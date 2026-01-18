import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def createDOMBuilder(self, mode, schemaType):
    if schemaType is not None:
        raise xml.dom.NotSupportedErr('schemaType not yet supported')
    if mode == self.MODE_SYNCHRONOUS:
        return DOMBuilder()
    if mode == self.MODE_ASYNCHRONOUS:
        raise xml.dom.NotSupportedErr('asynchronous builders are not supported')
    raise ValueError('unknown value for mode')