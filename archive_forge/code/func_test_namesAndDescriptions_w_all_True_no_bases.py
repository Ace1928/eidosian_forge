import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test_namesAndDescriptions_w_all_True_no_bases(self):
    from zope.interface.interface import Attribute
    from zope.interface.interface import fromFunction

    def _bar():
        """DOCSTRING"""
    ATTRS = {'foo': Attribute('Foo', ''), 'bar': fromFunction(_bar)}
    one = self._makeOne(attrs=ATTRS)
    self.assertEqual(sorted(one.namesAndDescriptions(all=False)), [('bar', ATTRS['bar']), ('foo', ATTRS['foo'])])