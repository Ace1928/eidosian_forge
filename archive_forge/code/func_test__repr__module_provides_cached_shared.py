import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test__repr__module_provides_cached_shared(self):
    from zope.interface.declarations import ModuleType
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    inst = self._makeOne(ModuleType, IFoo)
    inst._v_module_names += ('some.module',)
    inst._v_module_names += ('another.module',)
    self.assertEqual(repr(inst), "directlyProvides(('some.module', 'another.module'), IFoo)")