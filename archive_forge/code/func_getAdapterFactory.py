from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def getAdapterFactory(fromInterface, toInterface, default):
    """Return registered adapter for a given class and interface.

    Note that is tied to the *Twisted* global registry, and will
    thus not find adapters registered elsewhere.
    """
    self = globalRegistry
    if not isinstance(fromInterface, interface.InterfaceClass):
        fromInterface = declarations.implementedBy(fromInterface)
    factory = self.lookup1(fromInterface, toInterface)
    if factory is None:
        factory = default
    return factory