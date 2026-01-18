from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def locateAdapterClass(self, klass, interfaceClass, default):
    return getAdapterFactory(klass, interfaceClass, default)