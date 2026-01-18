import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.get_object_properties')
def test_get_object_properties_dict(self, mock_obj_prop):
    expected_prop_dict = {'name': 'vm01'}
    mock_obj_content = mock.Mock()
    prop = mock.Mock()
    prop.name = 'name'
    prop.val = 'vm01'
    mock_obj_content.propSet = [prop]
    del mock_obj_content.missingSet
    mock_obj_prop.return_value = [mock_obj_content]
    vim = mock.Mock()
    moref = mock.Mock()
    res = vim_util.get_object_properties_dict(vim, moref, None)
    self.assertEqual(expected_prop_dict, res)