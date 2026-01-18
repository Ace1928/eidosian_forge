from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_addAdapter(self):
    """
        C{Componentized.setAdapter} adapts the instance by wrapping it with
        given adapter class, then stores it using C{addComponent}.
        """
    componentized = components.Componentized()
    componentized.addAdapter(Adept, ignoreClass=True)
    component = componentized.getComponent(IAdept)
    self.assertEqual(component.original, componentized)
    self.assertIsInstance(component, Adept)