import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def setIdAttributeNode(self, idAttr):
    if idAttr is None or not self.isSameNode(idAttr.ownerElement):
        raise xml.dom.NotFoundErr()
    if _get_containing_entref(self) is not None:
        raise xml.dom.NoModificationAllowedErr()
    if not idAttr._is_id:
        idAttr._is_id = True
        self._magic_id_nodes += 1
        self.ownerDocument._magic_id_count += 1
        _clear_id_cache(self)