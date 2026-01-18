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
@method(function('collection', nargs=(0, 1), sequence_types=('xs:string?', 'node()*')))
def evaluate_collection_function(self, context=None):
    if self.context is not None:
        context = self.context
    uri = self.get_argument(context)
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        return []
    elif not self or uri is None:
        if context.default_collection is None:
            raise self.error('FODC0002', 'no default collection has been defined')
        collection = context.default_collection
        sequence_type = self.parser.default_collection_type
    else:
        uri = self.get_absolute_uri(uri)
        try:
            collection = context.collections[uri]
        except (KeyError, TypeError):
            if is_local_dir_url(uri):
                raise self.error('FODC0004', 'collection URI is a directory')
            raise self.error('FODC0002', '{!r} collection not found'.format(uri)) from None
        try:
            sequence_type = self.parser.collection_types[uri]
        except (KeyError, TypeError):
            return collection
    if not match_sequence_type(collection, sequence_type, self.parser):
        msg = f'Type does not match sequence type {sequence_type!r}'
        raise self.error('XPDY0050', msg)
    return collection