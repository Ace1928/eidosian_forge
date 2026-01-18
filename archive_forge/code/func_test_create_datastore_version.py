from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
def test_create_datastore_version(self):
    image_id = uuidutils.generate_uuid()
    args = ['new_name', 'ds_name', 'ds_manager', image_id, '--active', '--default', '--image-tags', 'trove,mysql']
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.dsversion_mgmt_client.create.assert_called_once_with('new_name', 'ds_name', 'ds_manager', image_id, active='true', default='true', image_tags=['trove', 'mysql'], version=None)