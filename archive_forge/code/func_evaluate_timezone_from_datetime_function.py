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
@method(function('timezone-from-dateTime', nargs=1, sequence_types=('xs:dateTime?', 'xs:dayTimeDuration?')))
def evaluate_timezone_from_datetime_function(self, context=None):
    cls = DateTime if self.parser.xsd_version == '1.1' else DateTime10
    item = self.get_argument(self.context or context, cls=cls)
    if item is None or item.tzinfo is None:
        return []
    return DayTimeDuration(seconds=item.tzinfo.offset.total_seconds())