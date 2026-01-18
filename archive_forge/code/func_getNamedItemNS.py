import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def getNamedItemNS(self, namespaceURI, localName):
    for n in self._seq:
        if n.namespaceURI == namespaceURI and n.localName == localName:
            return n