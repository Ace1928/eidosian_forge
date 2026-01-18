from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def setPrefix(self, p, u=None):
    """
        Set the element namespace prefix.

        @param p: A new prefix for the element.
        @type p: basestring
        @param u: A namespace URI to be mapped to the prefix.
        @type u: basestring
        @return: self
        @rtype: L{Element}

        """
    self.prefix = p
    if p is not None and u is not None:
        self.expns = None
        self.addPrefix(p, u)
    return self