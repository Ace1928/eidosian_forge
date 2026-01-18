import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_instance_conforms_to_IAdapterRegistration(self):
    from zope.interface.interfaces import IAdapterRegistration
    from zope.interface.verify import verifyObject
    ar, _, _ = self._makeOne()
    verifyObject(IAdapterRegistration, ar)