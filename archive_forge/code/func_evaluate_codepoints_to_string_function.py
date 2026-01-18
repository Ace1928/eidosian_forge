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
@method(function('codepoints-to-string', nargs=1, sequence_types=('xs:integer*', 'xs:string')))
def evaluate_codepoints_to_string_function(self, context=None):
    if self.context is not None:
        context = self.context
    result = []
    for value in self[0].select(context):
        if isinstance(value, UntypedAtomic):
            value = int(value)
        if not isinstance(value, int):
            msg = 'invalid type {} for codepoint {}'.format(type(value), value)
            if isinstance(value, str):
                raise self.error('XPTY0004', msg)
            raise self.error('FORG0006', msg)
        elif is_xml_codepoint(value):
            result.append(chr(value))
        else:
            msg = '{} is not a valid XML 1.0 codepoint'.format(value)
            raise self.error('FOCH0001', msg)
    return ''.join(result)