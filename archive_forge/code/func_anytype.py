from suds import *
from suds.reader import DocumentReader
from suds.sax import Namespace
from suds.transport import TransportError
from suds.xsd import *
from suds.xsd.query import *
from suds.xsd.sxbase import *
from urllib.parse import urljoin
from logging import getLogger
def anytype(self):
    """Create an xsd:anyType reference."""
    p, u = Namespace.xsdns
    mp = self.root.findPrefix(u)
    if mp is None:
        mp = p
        self.root.addPrefix(p, u)
    return ':'.join((mp, 'anyType'))