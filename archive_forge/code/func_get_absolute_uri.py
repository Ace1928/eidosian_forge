import decimal
import math
from copy import copy
from decimal import Decimal
from itertools import product
from typing import TYPE_CHECKING, cast, Dict, Optional, List, Tuple, \
import urllib.parse
from .exceptions import ElementPathError, ElementPathValueError, \
from .helpers import ordinal, get_double, split_function_test
from .etree import is_etree_element, is_etree_document
from .namespaces import XSD_NAMESPACE, XPATH_FUNCTIONS_NAMESPACE, \
from .tree_builders import get_node_tree
from .xpath_nodes import XPathNode, ElementNode, AttributeNode, \
from .datatypes import xsd10_atomic_types, AbstractDateTime, AnyURI, \
from .protocols import ElementProtocol, DocumentProtocol, XsdAttributeProtocol, \
from .sequence_types import is_sequence_type_restriction, match_sequence_type
from .schema_proxy import AbstractSchemaProxy
from .tdop import Token, MultiLabel
from .xpath_context import XPathContext, XPathSchemaContext
def get_absolute_uri(self, uri: str, base_uri: Optional[str]=None, as_string: bool=True) -> Union[str, AnyURI]:
    """
        Obtains an absolute URI from the argument and the static context.

        :param uri: a string representing a URI.
        :param base_uri: an alternative base URI, otherwise the base_uri         of the static context is used.
        :param as_string: if `True` then returns the URI as a string, otherwise         returns the URI as xs:anyURI instance.
        :returns: the argument if it's an absolute URI, otherwise returns the URI
        obtained by the join o the base_uri of the static context with the
        argument. Returns the argument if the base_uri is `None`.
        """
    if not base_uri:
        base_uri = self.parser.base_uri
    uri_parts: urllib.parse.ParseResult = urllib.parse.urlparse(uri)
    if uri_parts.scheme or uri_parts.netloc or base_uri is None:
        return uri if as_string else AnyURI(uri)
    base_uri_parts: urllib.parse.SplitResult = urllib.parse.urlsplit(base_uri)
    if base_uri_parts.fragment or (not base_uri_parts.scheme and (not base_uri_parts.netloc) and (not base_uri_parts.path.startswith('/'))):
        raise self.error('FORG0002', '{!r} is not suitable as base URI'.format(base_uri))
    if uri_parts.path.startswith('/') and base_uri_parts.path not in ('', '/'):
        return uri if as_string else AnyURI(uri)
    if as_string:
        return urllib.parse.urljoin(base_uri, uri)
    return AnyURI(urllib.parse.urljoin(base_uri, uri))