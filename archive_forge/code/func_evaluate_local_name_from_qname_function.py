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
@method(function('local-name-from-QName', nargs=1, sequence_types=('xs:QName?', 'xs:NCName?')))
def evaluate_local_name_from_qname_function(self, context=None):
    if self.context is not None:
        context = self.context
    qname = self.get_argument(context)
    if qname is None:
        return []
    elif not isinstance(qname, QName):
        if self.parser.version >= '3.0' and isinstance(self.data_value(qname), UntypedAtomic):
            code = 'XPTY0117'
        else:
            code = 'XPTY0004'
        raise self.error(code, 'argument has an invalid type %r' % type(qname))
    return NCName(qname.local_name)