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
def json_object_to_etree(obj):
    keys = set()
    items = []
    for k, v in obj:
        if k not in keys:
            keys.add(k)
        elif duplicates == 'use-first':
            continue
        elif duplicates == 'reject':
            raise self.error('FOJS0003')
        if not escape:
            k = ''.join((x if is_xml_codepoint(ord(x)) else fallback(f'\\u{ord(x):04X}', context=context) for x in k))
            k = k.replace('"', '&#34;')
            attrib = {'key': k}
        else:
            k = escape_string(k)
            if '\\' in k:
                attrib = {'escaped-key': 'true', 'key': k}
            else:
                attrib = {'key': k}
        items.append(value_to_etree(v, **attrib))
    elem = etree.Element(MAP_TAG)
    for item in items:
        elem.append(item)
    return elem