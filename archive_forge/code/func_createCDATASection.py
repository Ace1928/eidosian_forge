import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def createCDATASection(self, data):
    if not isinstance(data, str):
        raise TypeError('node contents must be a string')
    c = CDATASection()
    c.data = data
    c.ownerDocument = self
    return c