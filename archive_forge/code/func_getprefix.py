from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def getprefix(self, u):
    """
        Get the prefix for the specified namespace (URI)
        @param u: A namespace URI.
        @type u: str
        @return: The namspace.
        @rtype: (prefix, uri).
        """
    for ns in Namespace.all:
        if u == ns[1]:
            return ns[0]
    for ns in self.prefixes:
        if u == ns[1]:
            return ns[0]
    raise Exception('ns (%s) not mapped' % u)