from suds import *
from suds.sax import *
from suds.sax.text import Text
from suds.sax.attribute import Attribute
def refitAddr(self, a):
    """
        Refit (normalize) the attribute.

        @param a: An attribute.
        @type a: L{Attribute}

        """
    if a.prefix is not None:
        ns = a.namespace()
        if self.permit(ns):
            a.prefix = self.prefixes[ns[1]]
    self.refitValue(a)