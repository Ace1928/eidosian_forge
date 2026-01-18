from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def endElementNS(self, ns_name, qname):
    element = self._element_stack.pop()
    el_tag = self._buildTag(ns_name)
    if el_tag != element.tag:
        raise SaxError('Unexpected element closed: ' + el_tag)