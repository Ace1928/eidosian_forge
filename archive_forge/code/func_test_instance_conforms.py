import unittest
def test_instance_conforms(self):
    from zope.interface.verify import verifyObject
    verifyObject(self._getTargetInterface(), self._makeOne())