import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def supportsFeature(self, name):
    return hasattr(self._options, _name_xform(name))