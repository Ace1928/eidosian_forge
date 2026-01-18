import unittest
from zope.interface.tests import CleanUp
from zope.interface.tests import MissingSomeAttrs
from zope.interface.tests import OptimizationTestMixin
def test___call___w_adapter_hook(self):
    from zope.interface import Interface
    from zope.interface.interface import adapter_hooks

    def _miss(iface, obj):
        pass

    def _hit(iface, obj):
        return self

    class I(Interface):
        pass

    class C:
        pass
    c = C()
    old_adapter_hooks = adapter_hooks[:]
    adapter_hooks[:] = [_miss, _hit]
    try:
        self.assertTrue(I(c) is self)
    finally:
        adapter_hooks[:] = old_adapter_hooks