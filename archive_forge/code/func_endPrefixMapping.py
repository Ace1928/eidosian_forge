from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def endPrefixMapping(self, prefix):
    ns_uri_list = self._ns_mapping[prefix]
    ns_uri_list.pop()
    if prefix is None:
        self._default_ns = ns_uri_list[-1]