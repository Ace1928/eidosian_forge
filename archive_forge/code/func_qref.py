from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
def qref(self):
    """
        Get the I{type} qualified reference to the referenced XSD type.

        This method takes into account simple types defined through restriction
        which are detected by determining that self is simple (len == 0) and by
        finding a restriction child.

        @return: The I{type} qualified reference.
        @rtype: qref

        """
    qref = self.type
    if qref is None and len(self) == 0:
        ls = []
        m = RestrictionMatcher()
        finder = NodeFinder(m, 1)
        finder.find(self, ls)
        if ls:
            return ls[0].ref
    return qref