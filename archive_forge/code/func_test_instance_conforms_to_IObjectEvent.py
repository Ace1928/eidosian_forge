import unittest
def test_instance_conforms_to_IObjectEvent(self):
    from zope.interface.interfaces import IObjectEvent
    from zope.interface.verify import verifyObject
    verifyObject(IObjectEvent, self._makeOne())