from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_attributeCustomization(self):
    """
        The original attribute name can be customized via the
        C{originalAttribute} argument of L{proxyForInterface}: the attribute
        should change, but the methods of the original object should still be
        callable, and the attributes still accessible.
        """
    yayable = Yayable()
    yayable.ifaceAttribute = object()
    proxy = proxyForInterface(IProxiedInterface, originalAttribute='foo')(yayable)
    self.assertIs(proxy.foo, yayable)
    self.assertEqual(proxy.yay(), 1)
    self.assertIs(proxy.ifaceAttribute, yayable.ifaceAttribute)
    thingy = object()
    proxy.ifaceAttribute = thingy
    self.assertIs(yayable.ifaceAttribute, thingy)
    del proxy.ifaceAttribute
    self.assertFalse(hasattr(yayable, 'ifaceAttribute'))