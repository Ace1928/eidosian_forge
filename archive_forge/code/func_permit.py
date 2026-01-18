from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def permit(self, ns):
    """
        Get whether the I{ns} is to be normalized.

        @param ns: A namespace.
        @type ns: (p, u)
        @return: True if to be included.
        @rtype: boolean

        """
    return not self.skip(ns)