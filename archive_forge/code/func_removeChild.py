import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def removeChild(self, oldChild):
    try:
        self.childNodes.remove(oldChild)
    except ValueError:
        raise xml.dom.NotFoundErr()
    oldChild.nextSibling = oldChild.previousSibling = None
    oldChild.parentNode = None
    if self.documentElement is oldChild:
        self.documentElement = None
    return oldChild