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
@method(function('find', prefix='map', nargs=2, sequence_types=('map(*)', 'xs:anyAtomicType', 'array(*)')))
def evaluate_map_find_function(self, context=None):
    if self.context is not None:
        context = self.context
    key = self.get_argument(context, index=1, required=True, cls=AnyAtomicType)
    items = []

    def iter_matching_items(obj):
        if isinstance(obj, list):
            for x in obj:
                iter_matching_items(x)
        elif isinstance(obj, XPathArray):
            for x in obj.items(context):
                iter_matching_items(x)
        elif isinstance(obj, XPathMap):
            for k, v in obj.items(context):
                if k == key:
                    items.append(v)
                iter_matching_items(v)
    for item in self[0].select(context):
        iter_matching_items(item)
    return XPathArray(self.parser, items)