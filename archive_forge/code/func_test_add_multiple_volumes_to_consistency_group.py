from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import consistency_group
def test_add_multiple_volumes_to_consistency_group(self):
    volumes = volume_fakes.create_volumes(count=2)
    self.volumes_mock.get = volume_fakes.get_volumes(volumes)
    arglist = [self._consistency_group.id, volumes[0].id, volumes[1].id]
    verifylist = [('consistency_group', self._consistency_group.id), ('volumes', [volumes[0].id, volumes[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'add_volumes': volumes[0].id + ',' + volumes[1].id}
    self.consistencygroups_mock.update.assert_called_once_with(self._consistency_group.id, **kwargs)
    self.assertIsNone(result)