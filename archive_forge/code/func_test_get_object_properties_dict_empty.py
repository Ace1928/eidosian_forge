import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.get_object_properties')
def test_get_object_properties_dict_empty(self, mock_obj_prop):
    mock_obj_prop.return_value = None
    vim = mock.Mock()
    moref = mock.Mock()
    res = vim_util.get_object_properties_dict(vim, moref, None)
    self.assertEqual({}, res)