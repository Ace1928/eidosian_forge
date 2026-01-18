from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def createParser(self):
    """Create a new namespace-handling parser."""
    parser = expat.ParserCreate(namespace_separator=' ')
    parser.namespace_prefixes = True
    return parser