import copy
from keystoneauth1.exceptions import http as ksa_exceptions
from osc_lib import exceptions
from openstackclient.identity.v3 import limit
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_limit_delete(self):
    self.limit_mock.delete.return_value = None
    arglist = [identity_fakes.limit_id]
    verifylist = [('limit_id', [identity_fakes.limit_id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.limit_mock.delete.assert_called_with(identity_fakes.limit_id)
    self.assertIsNone(result)