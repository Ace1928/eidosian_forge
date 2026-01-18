from unittest.mock import call
from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_message
def test_message_list_with_options(self):
    self.volume_client.api_version = api_versions.APIVersion('3.3')
    arglist = ['--project', self.fake_project.name, '--marker', self.fake_messages[0].id, '--limit', '3']
    verifylist = [('project', self.fake_project.name), ('marker', self.fake_messages[0].id), ('limit', 3)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    search_opts = {'project_id': self.fake_project.id}
    self.volume_messages_mock.list.assert_called_with(search_opts=search_opts, marker=self.fake_messages[0].id, limit=3)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))