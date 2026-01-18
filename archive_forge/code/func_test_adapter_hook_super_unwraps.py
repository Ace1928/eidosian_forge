import unittest
from zope.interface.tests import OptimizationTestMixin
def test_adapter_hook_super_unwraps(self):
    _f_called_with = []

    def _factory(context):
        _f_called_with.append(context)
        return context

    def _lookup(self, required, provided, name=''):
        return _factory
    required = super()
    provided = object()
    lb = self._makeOne(uc_lookup=_lookup)
    adapted = lb.adapter_hook(provided, required)
    self.assertIs(adapted, self)
    self.assertEqual(_f_called_with, [self])