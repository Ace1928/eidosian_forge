from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
@mock.patch.object(utils, 'find_resource')
def test_instance_create_nic_param(self, mock_find):
    fake_id = self.random_uuid()
    mock_find.return_value.id = fake_id
    args = ['test-mysql', '--flavor', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d', '--size', '1', '--datastore', 'mysql', '--datastore-version', '5.7.29', '--nic', 'net-id=net1,subnet-id=subnet_id,ip-address=192.168.1.11']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.instance_client.create.assert_called_once_with('test-mysql', flavor_id=fake_id, volume={'size': 1, 'type': None}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore='mysql', datastore_version='5.7.29', datastore_version_number=None, nics=[{'network_id': 'net1', 'subnet_id': 'subnet_id', 'ip_address': '192.168.1.11'}], configuration=None, replica_of=None, replica_count=None, modules=[], locality=None, region_name=None, access={'is_public': False})