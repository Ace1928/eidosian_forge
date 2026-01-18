import unittest
def test_instance_conforms_to_IRegistered(self):
    from zope.interface.interfaces import IRegistered
    from zope.interface.verify import verifyObject
    verifyObject(IRegistered, self._makeOne())