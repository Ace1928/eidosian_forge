import unittest
from zope.interface.tests import OptimizationTestMixin
def test_adapter_hook_hit_factory_returns_adapter(self):
    _f_called_with = []
    _adapter = object()

    def _factory(context):
        _f_called_with.append(context)
        return _adapter

    def _lookup(self, required, provided, name):
        return _factory
    req, prv, _default = (object(), object(), object())
    lb = self._makeOne(uc_lookup=_lookup)
    adapted = lb.adapter_hook(prv, req, 'C', _default)
    self.assertIs(adapted, _adapter)
    self.assertEqual(_f_called_with, [req])