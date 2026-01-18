import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_instance_conforms_to_IHandlerRegistration(self):
    from zope.interface.interfaces import IHandlerRegistration
    from zope.interface.verify import verifyObject
    hr, _, _ = self._makeOne()
    verifyObject(IHandlerRegistration, hr)