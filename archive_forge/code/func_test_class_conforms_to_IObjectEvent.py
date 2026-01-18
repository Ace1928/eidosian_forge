import unittest
def test_class_conforms_to_IObjectEvent(self):
    from zope.interface.interfaces import IObjectEvent
    from zope.interface.verify import verifyClass
    verifyClass(IObjectEvent, self._getTargetClass())