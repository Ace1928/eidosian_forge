import json
import locale
import math
import pathlib
import random
import re
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import product
from urllib.request import urlopen
from urllib.parse import urlsplit
from ..datatypes import AnyAtomicType, AbstractBinary, AbstractDateTime, \
from ..exceptions import ElementPathTypeError
from ..helpers import WHITESPACES_PATTERN, is_xml_codepoint, \
from ..namespaces import XPATH_FUNCTIONS_NAMESPACE, XML_BASE
from ..etree import etree_iter_strings, is_etree_element
from ..collations import CollationManager
from ..compare import get_key_function, same_key
from ..tree_builders import get_node_tree
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode
from ..xpath_tokens import XPathFunction, XPathMap, XPathArray
from ..xpath_context import XPathSchemaContext
from ..validators import validate_json_to_xml
from ._xpath31_operators import XPath31Parser
@method(function('json-doc', nargs=(1, 2), sequence_types=('xs:string?', 'map(*)', 'item()?')))
@method(function('parse-json', nargs=(1, 2), sequence_types=('xs:string?', 'map(*)', 'item()?')))
def evaluate_parse_json_functions(self, context=None):
    if self.symbol == 'json-doc':
        href = self.get_argument(context, cls=str)
        if href is None:
            return []
        try:
            if urlsplit(href).scheme:
                with urlopen(href) as fp:
                    json_text = fp.read().decode('utf-8')
            else:
                with pathlib.Path(href).open() as fp:
                    json_text = fp.read()
        except IOError:
            raise self.error('FOUT1170') from None
    else:
        href = None
        json_text = self.get_argument(context, cls=str)
        if json_text is None:
            return []

    def _fallback(*_args, **_kwargs):
        return '�'
    liberal = False
    duplicates = 'use-first'
    escape = None
    fallback = _fallback
    if len(self) > 1:
        map_ = self.get_argument(context, index=1, required=True, cls=XPathMap)
        for k, v in map_.items(context):
            if k == 'liberal':
                if not isinstance(v, bool):
                    raise self.error('XPTY0004')
                liberal = v
            elif k == 'duplicates':
                if not isinstance(v, str):
                    raise self.error('XPTY0004')
                elif v not in ('use-first', 'use-last', 'reject'):
                    raise self.error('FOJS0005')
                duplicates = v
            elif k == 'escape':
                if not isinstance(v, bool):
                    raise self.error('XPTY0004')
                escape = v
            elif k == 'fallback':
                if not isinstance(v, XPathFunction):
                    msg = 'fallback parameter is not a function'
                    raise self.error('XPTY0004', msg)
                elif v.arity != 1:
                    msg = f'fallback function has arity {v.arity} (must be 1)'
                    raise self.error('XPTY0004', msg)
                elif escape:
                    msg = "cannot provide both 'fallback' and 'escape' parameters"
                    raise self.error('FOJS0005', msg)
                fallback = v
                escape = False

    def decode_value(value):
        if value is None:
            return []
        elif isinstance(value, list):
            return XPathArray(self.parser, [decode_value(x) for x in value])
        elif not isinstance(value, str):
            return value
        elif escape:
            return json.dumps(value, ensure_ascii=True)[1:-1].replace('\\"', '"')
        return ''.join((x if is_xml_codepoint(ord(x)) else fallback(f'\\u{ord(x):04X}', context=context) for x in value))

    def json_object_pairs_to_map(obj):
        items = {}
        for k_, v_ in obj:
            k_, v_ = (decode_value(k_), decode_value(v_))
            if k_ in items:
                if duplicates == 'use-first':
                    continue
                elif duplicates == 'reject':
                    raise self.error('FOJS0003')
            if isinstance(v_, list):
                values = [decode_value(x) for x in v_]
                items[k_] = XPathArray(self.parser, values) if values else values
            else:
                items[k_] = v_
        return XPathMap(self.parser, items)
    kwargs = {'object_pairs_hook': json_object_pairs_to_map}
    if liberal or escape:
        kwargs['strict'] = False
    if liberal:

        def parse_constant(s):
            raise self.error('FOJS0001')
        kwargs['parse_constant'] = parse_constant
    try:
        result = json.JSONDecoder(**kwargs).decode(json_text)
    except json.JSONDecodeError:
        if href and urlsplit(href).fragment:
            raise self.error('FOUT1170') from None
        raise self.error('FOJS0001') from None
    else:
        return decode_value(result)