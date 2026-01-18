import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.get_object_properties')
def test_get_object_properties_dict_missing(self, mock_obj_prop):
    mock_obj_content = mock.Mock()
    missing_prop = mock.Mock()
    missing_prop.path = 'name'
    missing_prop.fault = mock.Mock()
    mock_obj_content.missingSet = [missing_prop]
    del mock_obj_content.propSet
    mock_obj_prop.return_value = [mock_obj_content]
    vim = mock.Mock()
    moref = mock.Mock()
    res = vim_util.get_object_properties_dict(vim, moref, None)
    self.assertEqual({}, res)