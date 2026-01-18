from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_interfaceInheritance(self):
    """
        Proxies of subinterfaces generated with proxyForInterface should allow
        access to attributes of both the child and the base interfaces.
        """
    proxyClass = proxyForInterface(IProxiedSubInterface)
    booable = Booable()
    proxy = proxyClass(booable)
    proxy.yay()
    proxy.boo()
    self.assertTrue(booable.yayed)
    self.assertTrue(booable.booed)