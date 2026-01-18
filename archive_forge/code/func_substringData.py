import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def substringData(self, offset, count):
    if offset < 0:
        raise xml.dom.IndexSizeErr('offset cannot be negative')
    if offset >= len(self.data):
        raise xml.dom.IndexSizeErr('offset cannot be beyond end of data')
    if count < 0:
        raise xml.dom.IndexSizeErr('count cannot be negative')
    return self.data[offset:offset + count]