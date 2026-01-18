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
@method(function('contains-token', nargs=(2, 3), sequence_types=('xs:string*', 'xs:string', 'xs:string', 'xs:boolean')))
def evaluate_contains_token_function(self, context=None):
    if self.context is not None:
        context = self.context
    token_string = self.get_argument(context, index=1, required=True, cls=str)
    token_string = token_string.strip()
    if len(self) < 3:
        collation = self.parser.default_collation
    else:
        collation = self.get_argument(context, 2, required=True, cls=str)
    with CollationManager(collation, self) as manager:
        for input_string in self[0].select(context):
            if not isinstance(input_string, str):
                raise self.error('XPTY0004')
            if any((x and manager.eq(token_string, x) for x in re.split('[ \t\n\r\x0c\x0b]+', input_string))):
                return True
        else:
            return False