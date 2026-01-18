from xml.sax.handler import ContentHandler
from lxml import etree
from lxml.etree import ElementTree, SubElement
from lxml.etree import Comment, ProcessingInstruction
def processingInstruction(self, target, data):
    pi = ProcessingInstruction(target, data)
    if self._root is None:
        self._root_siblings.append(pi)
    else:
        self._element_stack[-1].append(pi)