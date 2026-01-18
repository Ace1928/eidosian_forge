import decimal
import os
import re
import codecs
import math
from copy import copy
from itertools import zip_longest
from typing import cast, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen
from urllib.error import URLError
from ..exceptions import ElementPathError
from ..tdop import MultiLabel
from ..helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, \
from ..namespaces import get_expanded_name, split_expanded_name, \
from ..datatypes import xsd10_atomic_types, NumericProxy, QName, Date10, \
from ..sequence_types import is_sequence_type, match_sequence_type
from ..etree import defuse_xml, etree_iter_paths
from ..xpath_nodes import XPathNode, ElementNode, TextNode, AttributeNode, \
from ..tree_builders import get_node_tree
from ..xpath_tokens import XPathFunctionArgType, XPathToken, ValueToken, XPathFunction
from ..serialization import get_serialization_params, serialize_to_xml, serialize_to_json
from ..xpath_context import XPathContext, XPathSchemaContext
from ..regex import translate_pattern, RegexError
from ._xpath30_operators import XPath30Parser
from .xpath30_helpers import UNICODE_DIGIT_PATTERN, DECIMAL_DIGIT_PATTERN, \
@method(function('parse-xml-fragment', nargs=1, sequence_types=('xs:string?', 'document-node()?')))
def evaluate_parse_xml_fragment_function(self, context=None):
    if self.context is not None:
        context = self.context
    arg = self.get_argument(context, cls=str)
    if arg is None or isinstance(context, XPathSchemaContext):
        return []
    elif context is None:
        raise self.missing_context()
    if arg.startswith('<?xml '):
        xml_declaration, _, arg = arg[6:].partition('?>')
        xml_params = DECL_PARAM_PATTERN.findall(xml_declaration)
        if 'encoding' not in xml_params:
            raise self.error('FODC0006', "'encoding' argument is mandatory")
        for param in xml_params:
            if param not in ('version', 'encoding'):
                msg = f'unexpected parameter {param!r} in XML declaration'
                raise self.error('FODC0006', msg)
    if arg.lstrip().startswith('<!DOCTYPE'):
        raise self.error('FODC0006', '<!DOCTYPE is not allowed')
    etree = context.etree
    try:
        if self.parser.defuse_xml:
            root = etree.XML(defuse_xml(arg))
        else:
            root = etree.XML(arg)
    except etree.ParseError as err:
        try:
            dummy_element_node = get_node_tree(root=etree.XML(f'<document>{arg}</document>'), namespaces=self.parser.namespaces)
        except etree.ParseError:
            raise self.error('FODC0006', str(err)) from None
        else:
            return DocumentNode.from_element_node(dummy_element_node)
    else:
        return get_node_tree(root=etree.ElementTree(root), namespaces=self.parser.namespaces)