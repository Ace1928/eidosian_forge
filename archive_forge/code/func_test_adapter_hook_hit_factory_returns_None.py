import unittest
from zope.interface.tests import OptimizationTestMixin
def test_adapter_hook_hit_factory_returns_None(self):
    _f_called_with = []

    def _factory(context):
        _f_called_with.append(context)

    def _lookup(self, required, provided, name):
        return _factory
    req, prv, _default = (object(), object(), object())
    lb = self._makeOne(uc_lookup=_lookup)
    adapted = lb.adapter_hook(prv, req, 'C', _default)
    self.assertIs(adapted, _default)
    self.assertEqual(_f_called_with, [req])