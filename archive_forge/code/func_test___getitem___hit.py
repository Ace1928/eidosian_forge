import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___getitem___hit(self):
    from zope.interface.interface import Attribute
    from zope.interface.interface import fromFunction

    def _bar():
        """DOCSTRING"""
    ATTRS = {'foo': Attribute('Foo', ''), 'bar': fromFunction(_bar)}
    one = self._makeOne(attrs=ATTRS)
    self.assertEqual(one['foo'], ATTRS['foo'])
    self.assertEqual(one['bar'], ATTRS['bar'])