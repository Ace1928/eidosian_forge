from unittest import mock
from neutron_lib.objects import utils as obj_utils
from neutron_lib.tests import _base as base
def test_get_updatable_fields(self):
    mock_class = mock.Mock()
    mock_class.fields_no_update = [0, 2, 6]
    mock_fields = mock.Mock()
    mock_fields.copy.return_value = {k: k for k in range(7)}
    updatable = obj_utils.get_updatable_fields(mock_class, mock_fields)
    self.assertEqual([1, 3, 4, 5], sorted(list(updatable.keys())))