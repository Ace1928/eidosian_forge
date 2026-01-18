from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def getRegistry():
    """Returns the Twisted global
    C{zope.interface.adapter.AdapterRegistry} instance.
    """
    return globalRegistry