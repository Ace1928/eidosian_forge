import sys
from . import server
from .workers import threadpool
from ._compat import ntob, bton
@classmethod
def gateway_map(cls):
    """Create a mapping of gateways and their versions.

        Returns:
            dict[tuple[int,int],class]: map of gateway version and
                corresponding class

        """
    return {gw.version: gw for gw in cls.__subclasses__()}