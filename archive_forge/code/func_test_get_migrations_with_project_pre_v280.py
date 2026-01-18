from unittest import mock
from novaclient import api_versions
from openstack import utils as sdk_utils
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_migration
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_get_migrations_with_project_pre_v280(self):
    self._set_mock_microversion('2.79')
    arglist = ['--status', 'migrating', '--changes-before', '2019-08-09T08:03:25Z', '--project', '0c2accde-644a-45fa-8c10-e76debc7fbc3']
    verifylist = [('status', 'migrating'), ('changes_before', '2019-08-09T08:03:25Z'), ('project', '0c2accde-644a-45fa-8c10-e76debc7fbc3')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    ex = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
    self.assertIn('--os-compute-api-version 2.80 or greater is required', str(ex))