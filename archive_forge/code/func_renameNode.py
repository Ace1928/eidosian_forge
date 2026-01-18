import io
import xml.dom
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE, domreg
from xml.dom.minicompat import *
from xml.dom.xmlbuilder import DOMImplementationLS, DocumentLS
def renameNode(self, n, namespaceURI, name):
    if n.ownerDocument is not self:
        raise xml.dom.WrongDocumentErr('cannot rename nodes from other documents;\nexpected %s,\nfound %s' % (self, n.ownerDocument))
    if n.nodeType not in (Node.ELEMENT_NODE, Node.ATTRIBUTE_NODE):
        raise xml.dom.NotSupportedErr('renameNode() only applies to element and attribute nodes')
    if namespaceURI != EMPTY_NAMESPACE:
        if ':' in name:
            prefix, localName = name.split(':', 1)
            if prefix == 'xmlns' and namespaceURI != xml.dom.XMLNS_NAMESPACE:
                raise xml.dom.NamespaceErr("illegal use of 'xmlns' prefix")
        else:
            if name == 'xmlns' and namespaceURI != xml.dom.XMLNS_NAMESPACE and (n.nodeType == Node.ATTRIBUTE_NODE):
                raise xml.dom.NamespaceErr("illegal use of the 'xmlns' attribute")
            prefix = None
            localName = name
    else:
        prefix = None
        localName = None
    if n.nodeType == Node.ATTRIBUTE_NODE:
        element = n.ownerElement
        if element is not None:
            is_id = n._is_id
            element.removeAttributeNode(n)
    else:
        element = None
    n.prefix = prefix
    n._localName = localName
    n.namespaceURI = namespaceURI
    n.nodeName = name
    if n.nodeType == Node.ELEMENT_NODE:
        n.tagName = name
    else:
        n.name = name
        if element is not None:
            element.setAttributeNode(n)
            if is_id:
                element.setIdAttributeNode(n)
    return n