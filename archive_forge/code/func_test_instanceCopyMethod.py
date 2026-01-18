import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_instanceCopyMethod(self) -> None:
    """
        Copying an instance method returns a new method with the same
        behavior.
        """
    foo = Foo()
    m = copy.copy(foo.method)
    self.assertEqual(m, foo.method)
    self.assertIsNot(m, foo.method)
    self.assertEqual('test-value', m())
    foo.instance_member = 'new-value'
    self.assertEqual('new-value', m())