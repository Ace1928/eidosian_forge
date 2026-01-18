import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def setUserData(self, key, data, handler):
    old = None
    try:
        d = self._user_data
    except AttributeError:
        d = {}
        self._user_data = d
    if key in d:
        old = d[key][0]
    if data is None:
        handler = None
        if old is not None:
            del d[key]
    else:
        d[key] = (data, handler)
    return old