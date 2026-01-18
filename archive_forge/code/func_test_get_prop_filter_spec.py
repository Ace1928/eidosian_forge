import collections
from unittest import mock
from oslo_vmware.tests import base
from oslo_vmware import vim_util
def test_get_prop_filter_spec(self):
    client_factory = mock.Mock()
    mock_obj = mock.Mock()
    filter_spec = vim_util.get_prop_filter_spec(client_factory, [mock_obj], ['test_prop'])
    self.assertEqual([mock_obj], filter_spec.objectSet)
    self.assertEqual(['test_prop'], filter_spec.propSet)