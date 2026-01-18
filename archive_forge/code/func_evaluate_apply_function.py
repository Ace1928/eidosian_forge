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
@method(function('apply', nargs=2, sequence_types=('function(*)', 'array(*)', 'item()*')))
def evaluate_apply_function(self, context=None):
    if self.context is not None:
        context = self.context
    if isinstance(self[0], XPathFunction):
        func = self[0]
    else:
        func = self.get_argument(context, required=True, cls=XPathFunction)
    array_ = self.get_argument(context, index=1, required=True, cls=XPathArray)
    try:
        return func(*array_.items(context), context=context)
    except ElementPathTypeError as err:
        if not err.code.endswith(('XPST0017', 'XPTY0004')):
            raise
        raise self.error('FOAP0001') from None