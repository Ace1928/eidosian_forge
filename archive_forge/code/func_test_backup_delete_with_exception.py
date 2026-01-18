from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient.osc.v1 import database_backups
from troveclient.tests.osc.v1 import fakes
@mock.patch('troveclient.utils.get_resource_id_by_name')
def test_backup_delete_with_exception(self, mock_getid):
    args = ['fakebackup']
    parsed_args = self.check_parser(self.cmd, args, [])
    mock_getid.side_effect = exceptions.CommandError
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)