from unittest import mock
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as v2_volume_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_manage
def test_block_storage_volume_manage_list__all_args(self):
    self.app.client_manager.volume.api_version = api_versions.APIVersion('3.8')
    arglist = ['fake_host', '--long', '--marker', 'fake_marker', '--limit', '5', '--offset', '3', '--sort', 'size:asc']
    verifylist = [('host', 'fake_host'), ('detailed', None), ('long', True), ('marker', 'fake_marker'), ('limit', '5'), ('offset', '3'), ('sort', 'size:asc')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    expected_columns = ['reference', 'size', 'safe_to_manage', 'reason_not_safe', 'cinder_id', 'extra_info']
    datalist = []
    for volume_record in self.volume_manage_list:
        manage_details = (volume_record.reference, volume_record.size, volume_record.safe_to_manage, volume_record.reason_not_safe, volume_record.cinder_id, volume_record.extra_info)
        datalist.append(manage_details)
    datalist = tuple(datalist)
    self.assertEqual(expected_columns, columns)
    self.assertEqual(datalist, tuple(data))
    self.volumes_mock.list_manageable.assert_called_with(host='fake_host', detailed=True, marker='fake_marker', limit='5', offset='3', sort='size:asc', cluster=None)