from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_address_scope_project_domain(self):
    project = identity_fakes_v3.FakeProject.create_one_project()
    self.projects_mock.get.return_value = project
    arglist = ['--project', project.id, '--project-domain', project.domain_id]
    verifylist = [('project', project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'project_id': project.id}
    self.network_client.address_scopes.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))