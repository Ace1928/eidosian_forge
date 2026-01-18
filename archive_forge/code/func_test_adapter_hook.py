import unittest
from zope.interface.tests import OptimizationTestMixin
def test_adapter_hook(self):
    a, b, _c = [object(), object(), object()]

    def _factory1(context):
        return a

    def _factory2(context):
        return b

    def _factory3(context):
        self.fail('This should never be called')
    _factories = [_factory1, _factory2, _factory3]

    def _lookup(self, required, provided, name):
        return _factories.pop(0)
    req, prv, _default = (object(), object(), object())
    reg = self._makeRegistry(3)
    lb = self._makeOne(reg, uc_lookup=_lookup)
    adapted = lb.adapter_hook(prv, req, 'C', _default)
    self.assertIs(adapted, a)
    adapted = lb.adapter_hook(prv, req, 'C', _default)
    self.assertIs(adapted, a)
    reg.ro[1]._generation += 1
    adapted = lb.adapter_hook(prv, req, 'C', _default)
    self.assertIs(adapted, b)