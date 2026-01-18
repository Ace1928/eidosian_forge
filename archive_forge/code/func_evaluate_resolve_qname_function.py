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
@method(function('resolve-QName', nargs=2, sequence_types=('xs:string?', 'element()', 'xs:QName?')))
def evaluate_resolve_qname_function(self, context=None):
    if self.context is not None:
        context = self.context
    qname = self.get_argument(context=copy(context))
    if qname is None:
        return []
    elif not isinstance(qname, str):
        raise self.error('FORG0006', '1st argument has an invalid type %r' % type(qname))
    if context is None:
        raise self.missing_context()
    elem = self.get_argument(context, index=1)
    if not isinstance(elem, ElementNode):
        raise self.error('FORG0006', '2nd argument %r is not an element node' % elem)
    qname = qname.strip()
    match = QNAME_PATTERN.match(qname)
    if match is None:
        raise self.error('FOCA0002', '1st argument must be an xs:QName')
    prefix = match.groupdict()['prefix'] or ''
    if prefix == 'xml':
        return QName(XML_NAMESPACE, qname)
    try:
        nsmap = elem.nsmap
    except AttributeError:
        nsmap = self.parser.namespaces
    for pfx, uri in nsmap.items():
        if pfx is None:
            pfx = ''
        if pfx == prefix:
            if pfx:
                return QName(uri, '{}:{}'.format(pfx, match.groupdict()['local']))
            else:
                return QName(uri, match.groupdict()['local'])
    if prefix or '' in nsmap or None in nsmap:
        raise self.error('FONS0004', 'no namespace found for prefix %r' % prefix)
    return QName('', qname)