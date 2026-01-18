import copy
from openstackclient.identity.v3 import implied_role
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_implied_role_create(self):
    arglist = [identity_fakes.ROLES[0]['id'], '--implied-role', identity_fakes.ROLES[1]['id']]
    verifylist = [('role', identity_fakes.ROLES[0]['id']), ('implied_role', identity_fakes.ROLES[1]['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.inference_rules_mock.create.assert_called_with(identity_fakes.ROLES[0]['id'], identity_fakes.ROLES[1]['id'])
    collist = ('implied', 'prior_role')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.ROLES[1]['id'], identity_fakes.ROLES[0]['id'])
    self.assertEqual(datalist, data)