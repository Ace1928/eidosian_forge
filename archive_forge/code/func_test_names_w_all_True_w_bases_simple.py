import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_names_w_all_True_w_bases_simple(self):
    from zope.interface.interface import Attribute
    from zope.interface.interface import fromFunction

    def _bar():
        """DOCSTRING"""
    BASE_ATTRS = {'foo': Attribute('Foo', ''), 'bar': fromFunction(_bar)}
    DERIVED_ATTRS = {'baz': Attribute('Baz', '')}
    base = self._makeOne('IBase', attrs=BASE_ATTRS)
    derived = self._makeOne('IDerived', bases=(base,), attrs=DERIVED_ATTRS)
    self.assertEqual(sorted(derived.names(all=True)), ['bar', 'baz', 'foo'])