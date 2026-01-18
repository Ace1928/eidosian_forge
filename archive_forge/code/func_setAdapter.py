from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def setAdapter(self, interfaceClass, adapterClass):
    """
        Cache a provider for the given interface, by adapting C{self} using
        the given adapter class.
        """
    self.setComponent(interfaceClass, adapterClass(self))