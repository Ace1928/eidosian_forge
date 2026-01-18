from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
def as_item(self) -> Tuple[Optional[str], str]:
    return (self.prefix, self.uri)