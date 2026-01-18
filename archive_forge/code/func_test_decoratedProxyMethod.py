from functools import wraps
from zope.interface import Attribute, Interface, implementer
from zope.interface.adapter import AdapterRegistry
from twisted.python import components
from twisted.python.compat import cmp, comparable
from twisted.python.components import _addHook, _removeHook, proxyForInterface
from twisted.trial import unittest
def test_decoratedProxyMethod(self):
    """
        Methods of the class created from L{proxyForInterface} can be used with
        the decorator-helper L{functools.wraps}.
        """
    base = proxyForInterface(IProxiedInterface)

    class klass(base):

        @wraps(base.yay)
        def yay(self):
            self.original.yays += 1
            return base.yay(self)
    original = Yayable()
    yayable = klass(original)
    yayable.yay()
    self.assertEqual(2, original.yays)