from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v1 import fakes as volume_fakes
from openstackclient.volume.v1 import volume_backup
def test_backup_restore_with_invalid_volume(self):
    arglist = [self.backup.id, 'unexist_volume']
    verifylist = [('backup', self.backup.id), ('volume', 'unexist_volume')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch.object(utils, 'find_resource', side_effect=exceptions.CommandError()):
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)