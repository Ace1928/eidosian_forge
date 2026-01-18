import unittest
def test_class_conforms_to_IRegistrationEvent(self):
    from zope.interface.interfaces import IRegistrationEvent
    from zope.interface.verify import verifyClass
    verifyClass(IRegistrationEvent, self._getTargetClass())