import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def cloneNode(self, deep):
    if not deep:
        return None
    clone = self.implementation.createDocument(None, None, None)
    clone.encoding = self.encoding
    clone.standalone = self.standalone
    clone.version = self.version
    for n in self.childNodes:
        childclone = _clone_node(n, deep, clone)
        assert childclone.ownerDocument.isSameNode(clone)
        clone.childNodes.append(childclone)
        if childclone.nodeType == Node.DOCUMENT_NODE:
            assert clone.documentElement is None
        elif childclone.nodeType == Node.DOCUMENT_TYPE_NODE:
            assert clone.doctype is None
            clone.doctype = childclone
        childclone.parentNode = clone
    self._call_user_data_handler(xml.dom.UserDataHandler.NODE_CLONED, self, clone)
    return clone