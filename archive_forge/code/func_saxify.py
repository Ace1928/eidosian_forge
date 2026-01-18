from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def saxify(self):
    self._content_handler.startDocument()
    element = self._element
    if hasattr(element, 'getprevious'):
        siblings = []
        sibling = element.getprevious()
        while getattr(sibling, 'tag', None) is ProcessingInstruction:
            siblings.append(sibling)
            sibling = sibling.getprevious()
        for sibling in siblings[::-1]:
            self._recursive_saxify(sibling, {})
    self._recursive_saxify(element, {})
    if hasattr(element, 'getnext'):
        sibling = element.getnext()
        while getattr(sibling, 'tag', None) is ProcessingInstruction:
            self._recursive_saxify(sibling, {})
            sibling = sibling.getnext()
    self._content_handler.endDocument()