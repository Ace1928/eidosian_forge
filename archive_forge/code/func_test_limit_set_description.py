import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_limit_set_description(self):
    limit = copy.deepcopy(identity_fakes.LIMIT)
    limit['description'] = identity_fakes.limit_description
    self.limit_mock.update.return_value = fakes.FakeResource(None, limit, loaded=True)
    arglist = ['--description', identity_fakes.limit_description, identity_fakes.limit_id]
    verifylist = [('description', identity_fakes.limit_description), ('limit_id', identity_fakes.limit_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.limit_mock.update.assert_called_with(identity_fakes.limit_id, description=identity_fakes.limit_description, resource_limit=None)
    collist = ('description', 'id', 'project_id', 'region_id', 'resource_limit', 'resource_name', 'service_id')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.limit_description, identity_fakes.limit_id, identity_fakes.project_id, None, identity_fakes.limit_resource_limit, identity_fakes.limit_resource_name, identity_fakes.service_id)
    self.assertEqual(datalist, data)