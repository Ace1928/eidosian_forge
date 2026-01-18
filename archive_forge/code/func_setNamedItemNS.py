import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def setNamedItemNS(self, node):
    raise xml.dom.NoModificationAllowedErr('NamedNodeMap instance is read-only')