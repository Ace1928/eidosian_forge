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
@method(function('avg', nargs=1, sequence_types=('xs:anyAtomicType*', 'xs:anyAtomicType')))
def evaluate_avg_function(self, context=None):
    if self.context is not None:
        context = self.context
    values = []
    for item in self[0].atomization(context):
        if isinstance(item, UntypedAtomic):
            values.append(self.cast_to_double(item.value))
        elif isinstance(item, (AnyURI, bool)):
            raise self.error('FORG0006', 'non numeric value {!r} in the sequence'.format(item))
        else:
            values.append(item)
    if not values:
        return []
    elif isinstance(values[0], Duration):
        value = values[0]
        try:
            for item in values[1:]:
                value = value + item
            return value / len(values)
        except TypeError as err:
            if isinstance(context, XPathSchemaContext):
                return []
            raise self.error('FORG0006', err)
    elif all((isinstance(x, int) for x in values)):
        result = sum(values) / Decimal(len(values))
        return int(result) if result % 1 == 0 else result
    elif all((isinstance(x, (int, Decimal)) for x in values)):
        return sum(values) / Decimal(len(values))
    elif all((not isinstance(x, DoubleProxy) for x in values)):
        try:
            return sum((Float10(x) if isinstance(x, Decimal) else x for x in values)) / len(values)
        except TypeError as err:
            if isinstance(context, XPathSchemaContext):
                return []
            raise self.error('FORG0006', err)
    else:
        try:
            return sum((float(x) if isinstance(x, Decimal) else x for x in values)) / len(values)
        except TypeError as err:
            if isinstance(context, XPathSchemaContext):
                return []
            raise self.error('FORG0006', err)