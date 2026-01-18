import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_obj_spec(self):
    client_factory = mock.Mock()
    mock_obj = mock.Mock()
    obj_spec = vim_util.get_obj_spec(client_factory, mock_obj, select_set=['abc'])
    self.assertEqual(mock_obj, obj_spec.obj)
    self.assertFalse(obj_spec.skip)
    self.assertEqual(['abc'], obj_spec.selectSet)