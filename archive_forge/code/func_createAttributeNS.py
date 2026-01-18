import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def createAttributeNS(self, namespaceURI, qualifiedName):
    prefix, localName = _nssplit(qualifiedName)
    a = Attr(qualifiedName, namespaceURI, localName, prefix)
    a.ownerDocument = self
    a.value = ''
    return a