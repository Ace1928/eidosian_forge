from importlib import import_module
from urllib.parse import urljoin
from types import ModuleType
from typing import cast, Any, Dict, Iterator, List, MutableMapping, Optional, Tuple, Union
from .datatypes import UntypedAtomic, get_atomic_value, AtomicValueType
from .namespaces import XML_NAMESPACE, XML_BASE, XSI_NIL, \
from .protocols import ElementProtocol, DocumentProtocol, XsdElementProtocol, \
from .helpers import match_wildcard, is_absolute_uri
from .etree import etree_iter_strings, is_etree_element, is_etree_document
@property
def namespace_nodes(self) -> List['NamespaceNode']:
    if self._namespace_nodes is None:
        position = self.position + 1
        self._namespace_nodes = [NamespaceNode('xml', XML_NAMESPACE, self, position)]
        position += 1
        if self.nsmap:
            for pfx, uri in self.nsmap.items():
                if pfx != 'xml':
                    self._namespace_nodes.append(NamespaceNode(pfx, uri, self, position))
                    position += 1
    return self._namespace_nodes