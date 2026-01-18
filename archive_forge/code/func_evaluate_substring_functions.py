import math
import datetime
import time
import re
import os.path
import unicodedata
from copy import copy
from decimal import Decimal, DecimalException
from string import ascii_letters
from urllib.parse import urlsplit, quote as urllib_quote
from ..exceptions import ElementPathValueError
from ..helpers import QNAME_PATTERN, is_idrefs, is_xml_codepoint, round_number
from ..datatypes import DateTime10, DateTime, Date10, Date, Float10, \
from ..namespaces import XML_NAMESPACE, get_namespace, split_expanded_name, \
from ..compare import deep_equal
from ..sequence_types import match_sequence_type
from ..xpath_context import XPathSchemaContext
from ..xpath_nodes import XPathNode, DocumentNode, ElementNode, SchemaElementNode
from ..xpath_tokens import XPathFunction
from ..regex import RegexError, translate_pattern
from ..collations import CollationManager
from ._xpath2_operators import XPath2Parser
@method(function('substring-before', nargs=(2, 3), sequence_types=('xs:string?', 'xs:string?', 'xs:string', 'xs:string')))
@method(function('substring-after', nargs=(2, 3), sequence_types=('xs:string?', 'xs:string?', 'xs:string', 'xs:string')))
def evaluate_substring_functions(self, context=None):
    if self.context is not None:
        context = self.context
    arg1 = self.get_argument(context, default='', cls=str)
    arg2 = self.get_argument(context, index=1, default='', cls=str)
    if len(self) < 3:
        collation = self.parser.default_collation
    else:
        collation = self.get_argument(context, 2, required=True, cls=str)
    with CollationManager(collation, self) as manager:
        index = manager.find(arg1, arg2)
    if index < 0:
        return ''
    if self.symbol == 'substring-before':
        return arg1[:index]
    else:
        return arg1[index + len(arg2):]