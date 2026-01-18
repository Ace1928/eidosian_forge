import unittest
def test_class_conforms_to_IUnregistered(self):
    from zope.interface.interfaces import IUnregistered
    from zope.interface.verify import verifyClass
    verifyClass(IUnregistered, self._getTargetClass())