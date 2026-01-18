from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor_profile
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
def test_create_with_metainfo(self):
    arglist = ['--description', self.new_flavor_profile.description, '--project', self.new_flavor_profile.project_id, '--project-domain', self.domain.name, '--enable', '--metainfo', self.new_flavor_profile.meta_info]
    verifylist = [('description', self.new_flavor_profile.description), ('project', self.new_flavor_profile.project_id), ('project_domain', self.domain.name), ('enable', True), ('metainfo', self.new_flavor_profile.meta_info)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_service_profile.assert_called_once_with(**{'description': self.new_flavor_profile.description, 'project_id': self.project.id, 'enabled': self.new_flavor_profile.is_enabled, 'metainfo': self.new_flavor_profile.meta_info})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)