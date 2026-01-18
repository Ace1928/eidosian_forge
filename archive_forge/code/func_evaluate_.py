from abc import ABCMeta
import locale
from collections.abc import MutableSequence
from urllib.parse import urlparse
from typing import cast, Any, Callable, ClassVar, Dict, List, \
from ..helpers import upper_camel_case, is_ncname, ordinal
from ..exceptions import ElementPathError, ElementPathTypeError, \
from ..namespaces import NamespacesType, XSD_NAMESPACE, XML_NAMESPACE, \
from ..collations import UNICODE_COLLATION_BASE_URI, UNICODE_CODEPOINT_COLLATION
from ..datatypes import UntypedAtomic, AtomicValueType, QName
from ..xpath_tokens import NargsType, XPathToken, ProxyToken, XPathFunction, XPathConstructor
from ..xpath_context import XPathContext, XPathSchemaContext
from ..sequence_types import is_sequence_type, match_sequence_type
from ..schema_proxy import AbstractSchemaProxy
from ..xpath1 import XPath1Parser
def evaluate_(self_: XPathFunction, context: Optional[XPathContext]=None) -> Union[List[None], AtomicValueType]:
    arg = self_.get_argument(context)
    if arg is None or self_.parser.schema is None:
        return []
    value = self_.string_value(arg)
    try:
        return self_.parser.schema.cast_as(value, atomic_type_name)
    except (TypeError, ValueError) as err:
        if isinstance(context, XPathSchemaContext):
            return []
        raise self_.error('FORG0001', err)