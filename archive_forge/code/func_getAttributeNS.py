import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def getAttributeNS(self, namespaceURI, localName):
    if self._attrsNS is None:
        return ''
    try:
        return self._attrsNS[namespaceURI, localName].value
    except KeyError:
        return ''