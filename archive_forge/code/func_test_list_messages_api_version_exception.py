from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import messages as osc_messages
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_messages_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.50')
    arglist = ['--before', '2021-02-06T09:49:58-05:00', '--since', '2021-02-05T09:49:58-05:00']
    verifylist = [('before', '2021-02-06T09:49:58-05:00'), ('since', '2021-02-05T09:49:58-05:00')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)