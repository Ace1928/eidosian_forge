from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
def test_instance_update_access(self):
    ins_id = '4c397f77-750d-43df-8fc5-f7388e4316ee'
    args = [ins_id, '--name', 'new_instance_name', '--is-private', '--allowed-cidr', '10.0.0.0/24', '--allowed-cidr', '10.0.1.0/24']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.instance_client.update.assert_called_with(ins_id, None, 'new_instance_name', False, False, is_public=False, allowed_cidrs=['10.0.0.0/24', '10.0.1.0/24'])