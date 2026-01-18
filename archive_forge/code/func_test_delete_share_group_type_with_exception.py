from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common.apiclient.exceptions import BadRequest
from manilaclient.common.apiclient.exceptions import NotFound
from manilaclient.osc import utils
from manilaclient.osc.v2 import share_group_types as osc_share_group_types
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_delete_share_group_type_with_exception(self):
    arglist = ['non_existing_type']
    verifylist = [('share_group_types', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.sgt_mock.delete.side_effect = exceptions.CommandError()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)