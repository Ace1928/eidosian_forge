import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
@mock.patch('oslo_vmware.vim_util.get_object_properties')
def test_get_object_property(self, get_object_properties):
    prop = mock.Mock()
    prop.val = 'ubuntu-12.04'
    properties = mock.Mock()
    properties.propSet = [prop]
    properties_list = [properties]
    get_object_properties.return_value = properties_list
    vim = mock.Mock()
    moref = mock.Mock()
    property_name = 'name'
    val = vim_util.get_object_property(vim, moref, property_name)
    self.assertEqual(prop.val, val)
    get_object_properties.assert_called_once_with(vim, moref, [property_name], skip_op_id=False)