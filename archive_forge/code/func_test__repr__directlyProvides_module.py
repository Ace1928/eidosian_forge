import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__directlyProvides_module(self):
    import sys
    from zope.interface.declarations import alsoProvides
    from zope.interface.declarations import directlyProvides
    from zope.interface.interface import InterfaceClass
    from zope.interface.tests import dummy
    IFoo = InterfaceClass('IFoo')
    IBar = InterfaceClass('IBar')
    orig_provides = dummy.__provides__
    del dummy.__provides__
    self.addCleanup(setattr, dummy, '__provides__', orig_provides)
    directlyProvides(dummy, IFoo)
    provides = dummy.__provides__
    self.assertEqual(repr(provides), "directlyProvides(sys.modules['zope.interface.tests.dummy'], IFoo)")
    alsoProvides(dummy, IBar)
    provides = dummy.__provides__
    self.assertEqual(repr(provides), "directlyProvides(sys.modules['zope.interface.tests.dummy'], IFoo, IBar)")
    my_module = sys.modules[__name__]
    assert not hasattr(my_module, '__provides__')
    directlyProvides(my_module, IFoo, IBar)
    self.addCleanup(delattr, my_module, '__provides__')
    self.assertIs(my_module.__provides__, provides)
    self.assertEqual(repr(provides), "directlyProvides(('zope.interface.tests.dummy', 'zope.interface.tests.test_declarations'), IFoo, IBar)")