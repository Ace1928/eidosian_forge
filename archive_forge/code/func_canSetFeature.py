import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def canSetFeature(self, name, state):
    key = (_name_xform(name), state and 1 or 0)
    return key in self._settings