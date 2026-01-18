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
@method(function('subarray', prefix='array', nargs=(2, 3), sequence_types=('array(*)', 'xs:integer', 'xs:integer', 'array(*)')))
def evaluate_array_subarray_function(self, context=None):
    if self.context is not None:
        context = self.context
    array_ = self.get_argument(context, required=True, cls=XPathArray)
    start = self.get_argument(context, index=1, required=True, cls=int)
    if start < 1 or start > len(array_) + 1:
        if isinstance(context, XPathSchemaContext):
            return array_
        raise self.error('FOAY0001')
    if len(self) > 2:
        length = self.get_argument(context, index=2, required=True, cls=int)
        if length < 0:
            raise self.error('FOAY0002')
        if start + length > len(array_) + 1:
            raise self.error('FOAY0001')
        items = array_.items(context)[start - 1:start + length - 1]
    else:
        items = array_.items(context)[start - 1:]
    return XPathArray(self.parser, items=items)