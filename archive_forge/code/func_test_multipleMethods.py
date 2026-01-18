from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_multipleMethods(self):
    """
        [Regression test] The proxy should send its method calls to the correct
        method, not the incorrect one.
        """
    multi = MultipleMethodImplementor()
    proxy = proxyForInterface(IMultipleMethods)(multi)
    self.assertEqual(proxy.methodOne(), 1)
    self.assertEqual(proxy.methodTwo(), 2)