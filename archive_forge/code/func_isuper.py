from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def isuper(self, iface, adapter):
    """
        Forward isuper to self.original
        """
    return self.original.isuper(iface, adapter)