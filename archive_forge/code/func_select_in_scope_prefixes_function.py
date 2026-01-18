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
@method(function('in-scope-prefixes', nargs=1, sequence_types=('element()', 'xs:string*')))
def select_in_scope_prefixes_function(self, context=None):
    if self.context is not None:
        context = self.context
    elif context is None:
        raise self.missing_context()
    arg = self.get_argument(context, required=True)
    if not isinstance(arg, ElementNode):
        raise self.error('XPTY0004', 'argument %r is not an element node' % arg)
    elem = arg.elem
    if isinstance(context, XPathSchemaContext):
        for pfx, uri in self.parser.namespaces.items():
            if uri:
                yield (pfx or '')
    elif hasattr(elem, 'nsmap'):
        if 'xml' not in elem.nsmap:
            yield 'xml'
        for pfx, uri in elem.nsmap.items():
            if uri:
                yield (pfx or '')
    else:
        for pfx, uri in self.parser.namespaces.items():
            if uri:
                yield (pfx or '')
        if context.namespaces:
            yield from (x for x in context.namespaces if x not in self.parser.namespaces)