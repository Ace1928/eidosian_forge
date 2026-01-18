import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_assignment_to__class__(self):

    class MyException(Exception):
        pass

    class MyInterfaceClass(self._getTargetClass()):

        def __call__(self, target):
            raise MyException(target)
    IFoo = self._makeOne('IName')
    self.assertIsInstance(IFoo, self._getTargetClass())
    self.assertIs(type(IFoo), self._getTargetClass())
    with self.assertRaises(TypeError):
        IFoo(1)
    IFoo.__class__ = MyInterfaceClass
    self.assertIsInstance(IFoo, MyInterfaceClass)
    self.assertIs(type(IFoo), MyInterfaceClass)
    with self.assertRaises(MyException):
        IFoo(1)