import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def test_instance_conforms_to_ISubscriptionAdapterRegistration(self):
    from zope.interface.interfaces import ISubscriptionAdapterRegistration
    from zope.interface.verify import verifyObject
    sar, _, _ = self._makeOne()
    verifyObject(ISubscriptionAdapterRegistration, sar)