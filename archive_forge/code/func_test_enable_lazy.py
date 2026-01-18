from oslotest import base as test_base
from oslo_i18n import _lazy
def test_enable_lazy(self):
    _lazy.USE_LAZY = False
    _lazy.enable_lazy()
    self.assertTrue(_lazy.USE_LAZY)