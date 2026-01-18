from unittest import mock
from stevedore import enabled
from neutron_lib.tests import _base as base
from neutron_lib.utils import runtime
@mock.patch.object(runtime.importutils, 'import_class', return_value=mock.sentinel.dummy_class)
@mock.patch.object(runtime, 'LOG')
def test_load_class_by_alias_or_classname_with_classname(self, mock_log, mock_import):
    self.assertEqual(mock.sentinel.dummy_class, runtime.load_class_by_alias_or_classname('ns', 'n'))