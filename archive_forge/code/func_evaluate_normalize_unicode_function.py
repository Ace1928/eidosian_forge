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
@method(function('normalize-unicode', nargs=(1, 2), sequence_types=('xs:string?', 'xs:string', 'xs:string')))
def evaluate_normalize_unicode_function(self, context=None):
    if self.context is not None:
        context = self.context
    arg = self.get_argument(context, default='', cls=str)
    if len(self) > 1:
        normalization_form = self.get_argument(context, 1, cls=str)
        if normalization_form is None:
            raise self.error('XPTY0004', "2nd argument can't be an empty sequence")
        else:
            normalization_form = normalization_form.strip().upper()
    else:
        normalization_form = 'NFC'
    if normalization_form == 'FULLY-NORMALIZED':
        msg = '%r normalization form not supported' % normalization_form
        raise self.error('FOCH0003', msg)
    if not arg:
        return ''
    elif not normalization_form:
        return arg
    try:
        return unicodedata.normalize(normalization_form, arg)
    except ValueError:
        msg = 'unsupported normalization form %r' % normalization_form
        raise self.error('FOCH0003', msg) from None