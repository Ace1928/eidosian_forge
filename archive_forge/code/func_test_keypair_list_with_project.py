import copy
from unittest import mock
from unittest.mock import call
import uuid
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.compute.v2 import keypair
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
@mock.patch.object(sdk_utils, 'supports_microversion', return_value=True)
def test_keypair_list_with_project(self, sm_mock):
    projects_mock = self.app.client_manager.identity.tenants
    projects_mock.reset_mock()
    projects_mock.get.return_value = fakes.FakeResource(None, copy.deepcopy(identity_fakes.PROJECT), loaded=True)
    users_mock = self.app.client_manager.identity.users
    users_mock.reset_mock()
    users_mock.list.return_value = [fakes.FakeResource(None, copy.deepcopy(identity_fakes.USER), loaded=True)]
    arglist = ['--project', identity_fakes.project_name]
    verifylist = [('project', identity_fakes.project_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    projects_mock.get.assert_called_with(identity_fakes.project_name)
    users_mock.list.assert_called_with(tenant_id=identity_fakes.project_id)
    self.compute_sdk_client.keypairs.assert_called_with(user_id=identity_fakes.user_id)
    self.assertEqual(('Name', 'Fingerprint', 'Type'), columns)
    self.assertEqual(((self.keypairs[0].name, self.keypairs[0].fingerprint, self.keypairs[0].type),), tuple(data))