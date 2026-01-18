import decimal
import os
import re
import codecs
import math
from copy import copy
from itertools import zip_longest
from typing import cast, Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit
from urllib.request import urlopen
from urllib.error import URLError
from ..exceptions import ElementPathError
from ..tdop import MultiLabel
from ..helpers import OCCURRENCE_INDICATORS, EQNAME_PATTERN, \
from ..namespaces import get_expanded_name, split_expanded_name, \
from ..datatypes import xsd10_atomic_types, NumericProxy, QName, Date10, \
from ..sequence_types import is_sequence_type, match_sequence_type
from ..etree import defuse_xml, etree_iter_paths
from ..xpath_nodes import XPathNode, ElementNode, TextNode, AttributeNode, \
from ..tree_builders import get_node_tree
from ..xpath_tokens import XPathFunctionArgType, XPathToken, ValueToken, XPathFunction
from ..serialization import get_serialization_params, serialize_to_xml, serialize_to_json
from ..xpath_context import XPathContext, XPathSchemaContext
from ..regex import translate_pattern, RegexError
from ._xpath30_operators import XPath30Parser
from .xpath30_helpers import UNICODE_DIGIT_PATTERN, DECIMAL_DIGIT_PATTERN, \
@method(function('uri-collection', nargs=(0, 1), sequence_types=('xs:string?', 'xs:anyURI*')))
def evaluate_uri_collection_function(self, context=None):
    if self.context is not None:
        context = self.context
    uri = self.get_argument(context)
    if context is None:
        raise self.missing_context()
    elif isinstance(context, XPathSchemaContext):
        return []
    elif not self or uri is None:
        if context.default_resource_collection is None:
            raise self.error('FODC0002', 'no default resource collection has been defined')
        resource_collection = AnyURI(context.default_resource_collection)
    else:
        try:
            AnyURI(uri)
        except ValueError:
            raise self.error('FODC0004', 'invalid argument to fn:uri-collection') from None
        uri = self.get_absolute_uri(uri)
        try:
            resource_collection = context.resource_collections[uri]
        except (KeyError, TypeError):
            url_parts = urlsplit(uri)
            if url_parts.scheme in ('', 'file') and (not url_parts.path.startswith(':')) and url_parts.path.endswith('/'):
                raise self.error('FODC0003', 'collection URI is a directory')
            raise self.error('FODC0002', '{!r} collection not found'.format(uri)) from None
    if not match_sequence_type(resource_collection, 'xs:anyURI*', self.parser):
        raise self.error('XPDY0050', 'Type does not match sequence type xs:anyURI*')
    return resource_collection