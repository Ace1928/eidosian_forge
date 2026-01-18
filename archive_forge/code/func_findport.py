from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def findport(self, port):
    """
        Find and return a port tuple for the specified port.
        Created and added when not found.
        @param port: A port.
        @type port: I{service.Port}
        @return: A port tuple.
        @rtype: (port, [method])
        """
    for p in self.ports:
        if p[0] == p:
            return p
    p = (port, [])
    self.ports.append(p)
    return p