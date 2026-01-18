from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def first_element_handler(self, name, attributes):
    if self._filter is None and (not self._elem_info):
        self._finish_end_element = id
    self.getParser().StartElementHandler = self.start_element_handler
    self.start_element_handler(name, attributes)