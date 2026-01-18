from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_multipleInterfacesForClass(self):
    """
        Test the registration of an adapter from a class to several
        interfaces at once.
        """

    class TheOriginal:
        pass
    return self._multipleInterfacesForClassOrInterface(TheOriginal)