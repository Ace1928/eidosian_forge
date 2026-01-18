import copy
import pickle
from twisted.persisted.styles import _UniversalPicklingError, unpickleMethod
from twisted.trial import unittest
def test_instanceBuildingNamePresent(self) -> None:
    """
        L{unpickleMethod} returns an instance method bound to the
        instance passed to it.
        """
    foo = Foo()
    m = unpickleMethod('method', foo, Foo)
    self.assertEqual(m, foo.method)
    self.assertIsNot(m, foo.method)