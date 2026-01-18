import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def parseURI(self, uri):
    if self.entityResolver:
        input = self.entityResolver.resolveEntity(None, uri)
    else:
        input = DOMEntityResolver().resolveEntity(None, uri)
    return self.parse(input)