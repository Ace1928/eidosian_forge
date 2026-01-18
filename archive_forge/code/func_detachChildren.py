from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def detachChildren(self):
    """
        Detach and return this element's children.

        @return: The element's children (detached).
        @rtype: [L{Element},...]

        """
    detached = self.children
    self.children = []
    for child in detached:
        child.parent = None
    return detached