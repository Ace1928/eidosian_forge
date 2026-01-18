import copy
from unittest import mock
import testtools
from ironicclient.common import base
from ironicclient import exc
from ironicclient.tests.unit import utils
@mock.patch.object(base.Manager, '_get', autospec=True)
def test__get_as_dict_empty(self, mock_get):
    mock_get.return_value = None
    resource_id = TESTABLE_RESOURCE['uuid']
    resource = self.manager._get_as_dict(resource_id)
    mock_get.assert_called_once_with(mock.ANY, resource_id, fields=None, os_ironic_api_version=None, global_request_id=None)
    self.assertEqual({}, resource)