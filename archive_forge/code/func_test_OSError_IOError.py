import unittest
def test_OSError_IOError(self):
    from zope.interface import providedBy
    from zope.interface.common import interfaces
    self.assertEqual(list(providedBy(OSError()).flattened()), [interfaces.IOSError, interfaces.IIOError, interfaces.IEnvironmentError, interfaces.IStandardError, interfaces.IException, interfaces.Interface])