import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_assignment_to__class__2(self):
    from zope.interface import Interface

    class MyInterfaceClass(self._getTargetClass()):

        def __call__(self, *args):
            return args
    IFoo = MyInterfaceClass('IFoo', (Interface,))
    self.assertEqual(IFoo(1), (1,))

    class IBar(IFoo):
        pass
    self.assertEqual(IBar(1), (1,))

    class ISpam(Interface):
        pass
    with self.assertRaises(TypeError):
        ISpam()
    ISpam.__class__ = MyInterfaceClass
    self.assertEqual(ISpam(1), (1,))