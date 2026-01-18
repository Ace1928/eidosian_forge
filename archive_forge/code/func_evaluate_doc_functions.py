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
@method(function('doc', nargs=1, sequence_types=('xs:string?', 'document-node()?')))
@method(function('doc-available', nargs=1, sequence_types=('xs:string?', 'xs:boolean')))
def evaluate_doc_functions(self, context=None):
    if self.context is not None:
        context = self.context
    uri = self.get_argument(context)
    if uri is None:
        return [] if self.symbol == 'doc' else False
    elif isinstance(uri, str):
        pass
    elif isinstance(uri, AnyURI):
        uri = str(uri)
    elif isinstance(uri, UntypedAtomic):
        raise self.error('FODC0002')
    else:
        raise self.error('XPTY0004')
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        return [] if self.symbol == 'doc' else False
    uri = uri.strip()
    if uri.startswith(':'):
        if self.symbol == 'doc' or self.parser.version <= '3.0':
            raise self.error('FODC0005')
        return False
    try:
        uri = self.get_absolute_uri(uri)
    except ElementPathValueError as err:
        if self.symbol == 'doc':
            raise self.error('FODC0002', err.message) from None
        return False
    try:
        doc = context.documents[uri]
    except (KeyError, TypeError):
        if self.symbol == 'doc':
            if is_local_dir_url(uri):
                raise self.error('FODC0005', 'document URI is a directory')
            raise self.error('FODC0002')
        return False
    else:
        if doc is None:
            raise self.error('FODC0002')
    try:
        sequence_type = self.parser.document_types[uri]
    except (KeyError, TypeError):
        sequence_type = 'document-node()'
    if not match_sequence_type(doc, sequence_type, self.parser):
        msg = f'Type does not match sequence type {sequence_type!r}'
        raise self.error('XPDY0050', msg)
    return doc if self.symbol == 'doc' else True