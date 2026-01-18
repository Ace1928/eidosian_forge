from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def refitAttrs(self, n):
    """
        Refit (normalize) all of the attributes in the node.

        @param n: A node.
        @type n: L{Element}

        """
    for a in n.attributes:
        self.refitAddr(a)