import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def createElementNS(self, namespaceURI, qualifiedName):
    prefix, localName = _nssplit(qualifiedName)
    e = Element(qualifiedName, namespaceURI, prefix)
    e.ownerDocument = self
    return e