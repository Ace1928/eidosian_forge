from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def setComponent(self, interfaceClass, component):
    """
        Cache a provider of the given interface.
        """
    self._adapterCache[reflect.qual(interfaceClass)] = component