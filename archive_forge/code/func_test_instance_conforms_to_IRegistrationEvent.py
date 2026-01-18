import unittest
def test_instance_conforms_to_IRegistrationEvent(self):
    from zope.interface.interfaces import IRegistrationEvent
    from zope.interface.verify import verifyObject
    verifyObject(IRegistrationEvent, self._makeOne())