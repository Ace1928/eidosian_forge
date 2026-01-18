from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_pools
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_pools_share_type_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.22')
    arglist = ['--share-type', self.share_type.id]
    verifylist = [('share_type', self.share_type.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)