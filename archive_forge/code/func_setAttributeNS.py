import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def setAttributeNS(self, namespaceURI, qualifiedName, value):
    prefix, localname = _nssplit(qualifiedName)
    attr = self.getAttributeNodeNS(namespaceURI, localname)
    if attr is None:
        attr = Attr(qualifiedName, namespaceURI, localname, prefix)
        attr.value = value
        attr.ownerDocument = self.ownerDocument
        self.setAttributeNode(attr)
    else:
        if value != attr.value:
            attr.value = value
            if attr.isId:
                _clear_id_cache(self)
        if attr.prefix != prefix:
            attr.prefix = prefix
            attr.nodeName = qualifiedName