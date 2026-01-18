import os
from ..exceptions import XMLSchemaException, XMLSchemaValueError
from ..names import XSD_NAMESPACE, WSDL_NAMESPACE, SOAP_NAMESPACE, \
from ..helpers import get_qname, local_name, get_extended_qname, get_prefixed_qname
from ..namespaces import NamespaceResourcesMap
from ..resources import fetch_resource
from ..documents import XmlDocument
from ..validators import XMLSchemaBase, XMLSchema10
def import_namespace(self, namespace, location, base_url=None):
    if namespace == self.target_namespace:
        msg = "namespace to import must be different from the 'targetNamespace' of the WSDL document"
        raise XMLSchemaValueError(msg)
    elif namespace in self.maps.imports:
        return self.maps.imports[namespace]
    url = fetch_resource(location, base_url or self.base_url)
    wsdl_document = self.__class__(source=url, maps=self.maps, namespaces=self._namespaces, validation=self.validation, base_url=self.base_url, allow=self.allow, defuse=self.defuse, timeout=self.timeout)
    if wsdl_document.target_namespace != namespace:
        msg = 'imported {!r} has an unmatched namespace {!r}'
        self.parse_error(msg.format(wsdl_document, namespace))
    self.maps.imports[namespace] = wsdl_document
    return wsdl_document