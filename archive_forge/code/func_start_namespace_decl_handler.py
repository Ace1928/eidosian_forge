from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def start_namespace_decl_handler(self, prefix, uri):
    """Push this namespace declaration on our storage."""
    self._ns_ordered_prefixes.append((prefix, uri))