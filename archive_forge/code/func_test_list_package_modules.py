from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
def test_list_package_modules(self):
    self.assertGreater(len(runtime.list_package_modules('neutron_lib.exceptions')), 3)