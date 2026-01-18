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
@method(function('year-from-date', nargs=1, sequence_types=('xs:date?', 'xs:integer?')))
@method(function('month-from-date', nargs=1, sequence_types=('xs:date?', 'xs:integer?')))
@method(function('day-from-date', nargs=1, sequence_types=('xs:date?', 'xs:integer?')))
@method(function('timezone-from-date', nargs=1, sequence_types=('xs:date?', 'xs:dayTimeDuration?')))
def evaluate_from_date_functions(self, context=None):
    cls = Date if self.parser.xsd_version == '1.1' else Date10
    item = self.get_argument(self.context or context, cls=cls)
    if item is None:
        return []
    elif self.symbol.startswith('year'):
        return item.year
    elif self.symbol.startswith('month'):
        return item.month
    elif self.symbol.startswith('day'):
        return item.day
    elif item.tzinfo is None:
        return []
    return DayTimeDuration(seconds=item.tzinfo.offset.total_seconds())