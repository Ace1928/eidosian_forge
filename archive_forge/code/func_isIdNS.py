import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def isIdNS(self, namespaceURI, localName):
    """Returns true iff the identified attribute is a DTD-style ID."""
    return False