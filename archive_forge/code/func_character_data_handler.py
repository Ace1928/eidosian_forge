from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def character_data_handler(self, data):
    childNodes = self.curNode.childNodes
    if childNodes and childNodes[-1].nodeType == TEXT_NODE:
        node = childNodes[-1]
        node.data = node.data + data
        return
    node = minidom.Text()
    node.data = node.data + data
    node.ownerDocument = self.document
    _append_child(self.curNode, node)