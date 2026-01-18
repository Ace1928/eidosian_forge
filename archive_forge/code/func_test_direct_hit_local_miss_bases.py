import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_direct_hit_local_miss_bases(self):
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
    self.assertEqual(derived.direct('foo'), DERIVED_ATTRS['foo'])
    self.assertEqual(derived.direct('baz'), DERIVED_ATTRS['baz'])
    self.assertEqual(derived.direct('bar'), None)