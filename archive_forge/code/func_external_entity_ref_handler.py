from a string or file.
from xml.dom import xmlbuilder, minidom, Node
from xml.dom import EMPTY_NAMESPACE, EMPTY_PREFIX, XMLNS_NAMESPACE
from xml.parsers import expat
from xml.dom.minidom import _append_child, _set_attribute_node
from xml.dom.NodeFilter import NodeFilter
def external_entity_ref_handler(self, context, base, systemId, publicId):
    if systemId == _FRAGMENT_BUILDER_INTERNAL_SYSTEM_ID:
        old_document = self.document
        old_cur_node = self.curNode
        parser = self._parser.ExternalEntityParserCreate(context)
        self.document = self.originalDocument
        self.fragment = self.document.createDocumentFragment()
        self.curNode = self.fragment
        try:
            parser.Parse(self._source, True)
        finally:
            self.curNode = old_cur_node
            self.document = old_document
            self._source = None
        return -1
    else:
        return ExpatBuilder.external_entity_ref_handler(self, context, base, systemId, publicId)