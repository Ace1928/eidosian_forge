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
def max_or_min():
    if not values:
        return values
    elif all((isinstance(x, str) for x in values)):
        if to_any_uri:
            return AnyURI(aggregate_func(values))
    elif any((isinstance(x, str) for x in values)):
        if any((isinstance(x, ArithmeticProxy) for x in values)):
            raise self.error('FORG0006', 'cannot compare strings with numeric data')
    elif all((isinstance(x, (Decimal, int)) for x in values)):
        return aggregate_func(values)
    elif any((isinstance(x, float) and math.isnan(x) for x in values)):
        return float_class('NaN')
    elif all((isinstance(x, (int, float, Decimal)) for x in values)):
        return float_class(aggregate_func(values))
    return aggregate_func(values)