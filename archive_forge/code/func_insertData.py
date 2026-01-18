import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def insertData(self, offset, arg):
    if offset < 0:
        raise xml.dom.IndexSizeErr('offset cannot be negative')
    if offset >= len(self.data):
        raise xml.dom.IndexSizeErr('offset cannot be beyond end of data')
    if arg:
        self.data = '%s%s%s' % (self.data[:offset], arg, self.data[offset:])