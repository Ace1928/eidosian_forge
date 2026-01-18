from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def testMultiAdapter(self):
    c = CComp()
    co1 = c.getComponent(ITest)
    co3 = c.getComponent(ITest3)
    co4 = c.getComponent(ITest4)
    self.assertIsNone(co4)
    self.assertIs(co1, co3)