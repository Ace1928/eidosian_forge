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
@method(function('abs', nargs=1, sequence_types=('xs:numeric?', 'xs:numeric?')))
def evaluate_abs_function(self, context=None):
    if self.context is not None:
        context = self.context
    item = self.get_argument(context)
    if item is None:
        return []
    elif isinstance(item, float) and math.isnan(item):
        return item
    elif isinstance(item, XPathNode):
        value = self.string_value(item)
        try:
            return abs(Decimal(value))
        except DecimalException:
            if isinstance(context, XPathSchemaContext):
                return []
            raise self.error('FOCA0002', 'invalid string value {!r} for {!r}'.format(value, item))
    elif isinstance(item, bool) or not isinstance(item, (float, int, Decimal)):
        raise self.error('XPTY0004', 'invalid argument type {!r}'.format(type(item)))
    else:
        return abs(item)