from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import datastores
from troveclient.tests.osc.v1 import fakes
def test_delete_datastore(self):
    ds_id = uuidutils.generate_uuid()
    args = [ds_id]
    parsed_args = self.check_parser(self.cmd, args, [])
    self.cmd.take_action(parsed_args)
    self.datastore_client.delete.assert_called_once_with(ds_id)