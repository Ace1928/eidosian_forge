from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet_pool
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_create_project_domain(self):
    arglist = ['--pool-prefix', '10.0.10.0/24', '--project', self.project.name, '--project-domain', self.domain.name, self._subnet_pool.name]
    verifylist = [('prefixes', ['10.0.10.0/24']), ('project', self.project.name), ('project_domain', self.domain.name), ('name', self._subnet_pool.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_subnet_pool.assert_called_once_with(**{'prefixes': ['10.0.10.0/24'], 'project_id': self.project.id, 'name': self._subnet_pool.name})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)