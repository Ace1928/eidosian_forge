from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def testAdapterGetComponent(self):
    o = object()
    a = Adept(o)
    self.assertRaises(components.CannotAdapt, ITest, a)
    self.assertIsNone(ITest(a, None))