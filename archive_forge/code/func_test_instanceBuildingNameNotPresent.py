import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_instanceBuildingNameNotPresent(self) -> None:
    """
        If the named method is not present in the class,
        L{unpickleMethod} finds a method on the class of the instance
        and returns a bound method from there.
        """
    foo = Foo()
    m = unpickleMethod('method', foo, Bar)
    self.assertEqual(m, foo.method)
    self.assertIsNot(m, foo.method)