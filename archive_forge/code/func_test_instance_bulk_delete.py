from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch('troveclient.utils.get_resource_id_by_name')
def test_instance_bulk_delete(self, mock_getid):
    instance_1 = uuidutils.generate_uuid()
    instance_2 = uuidutils.generate_uuid()
    mock_getid.return_value = instance_1
    args = ['fake_instance', instance_2]
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    mock_getid.assert_called_once_with(self.instance_client, 'fake_instance')
    calls = [mock.call(instance_1), mock.call(instance_2)]
    self.instance_client.delete.assert_has_calls(calls)