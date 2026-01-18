import unittest
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
from zope.interface.tests.test_interface import \
def test_called_from_class(self):
    from zope.interface.declarations import moduleProvides
    from zope.interface.interface import InterfaceClass
    IFoo = InterfaceClass('IFoo')
    globs = {'__name__': 'zope.interface.tests.foo', 'moduleProvides': moduleProvides, 'IFoo': IFoo}
    locs = {}
    CODE = '\n'.join(['class Foo(object):', '    moduleProvides(IFoo)'])
    with self.assertRaises(TypeError):
        exec(CODE, globs, locs)