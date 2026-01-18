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
@method(function('filter', prefix='array', nargs=2, sequence_types=('array(*)', 'function(item()*) as xs:boolean', 'array(*)')))
def evaluate_array_filter_function(self, context=None):
    if self.context is not None:
        context = self.context
    array_ = self.get_argument(context, required=True, cls=XPathArray)
    func = self.get_argument(context, index=1, required=True, cls=XPathFunction)
    items = array_.items(context)

    def filter_function(x):
        choice = func(x, context=context)
        if not isinstance(choice, bool):
            raise self.error('XPTY0004', f'{func} must return xs:boolean values')
        return choice
    return XPathArray(self.parser, items=filter(filter_function, items))