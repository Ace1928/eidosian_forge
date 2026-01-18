from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(runtime, 'LOG')
def test_load_class_by_alias_or_classname_no_name(self, mock_log):
    self.assertRaises(ImportError, runtime.load_class_by_alias_or_classname, 'ns', None)