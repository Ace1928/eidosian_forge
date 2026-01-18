import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_called_once_from_module_scope(self):
    from zope.interface.declarations import moduleProvides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    globs = {'__name__': 'zope.interface.tests.foo', 'moduleProvides': moduleProvides, 'IFoo': IFoo}
    CODE = '\n'.join(['moduleProvides(IFoo)'])
    exec(CODE, globs)
    spec = globs['__provides__']
    self.assertEqual(list(spec), [IFoo])