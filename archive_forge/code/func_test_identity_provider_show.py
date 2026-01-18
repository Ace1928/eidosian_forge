import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.identity.v3 import identity_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as test_utils
def test_identity_provider_show(self):
    arglist = [identity_fakes.idp_id]
    verifylist = [('identity_provider', identity_fakes.idp_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.identity_providers_mock.get.assert_called_with(identity_fakes.idp_id, id='test_idp')
    collist = ('description', 'domain_id', 'enabled', 'id', 'remote_ids')
    self.assertEqual(collist, columns)
    datalist = (identity_fakes.idp_description, identity_fakes.domain_id, True, identity_fakes.idp_id, identity_fakes.formatted_idp_remote_ids)
    self.assertCountEqual(datalist, data)