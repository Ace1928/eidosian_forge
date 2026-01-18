import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_namesAndDescriptions_w_all_True_bases_w_same_names(self):
    from zope.interface.interface import Attribute
    from zope.interface.interface import fromFunction

    def _bar():
        """DOCSTRING"""

    def _foo():
        """DOCSTRING"""
    BASE_ATTRS = {'foo': Attribute('Foo', ''), 'bar': fromFunction(_bar)}
    DERIVED_ATTRS = {'foo': fromFunction(_foo), 'baz': Attribute('Baz', '')}
    base = self._makeOne('IBase', attrs=BASE_ATTRS)
    derived = self._makeOne('IDerived', bases=(base,), attrs=DERIVED_ATTRS)
    self.assertEqual(sorted(derived.namesAndDescriptions(all=True)), [('bar', BASE_ATTRS['bar']), ('baz', DERIVED_ATTRS['baz']), ('foo', DERIVED_ATTRS['foo'])])