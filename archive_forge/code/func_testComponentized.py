from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def testComponentized(self):
    components.registerAdapter(Adept, Compo, IAdept)
    components.registerAdapter(Elapsed, Compo, IElapsed)
    c = Compo()
    assert c.getComponent(IAdept).adaptorFunc() == (1, 1)
    assert c.getComponent(IAdept).adaptorFunc() == (2, 2)
    assert IElapsed(IAdept(c)).elapsedFunc() == 1