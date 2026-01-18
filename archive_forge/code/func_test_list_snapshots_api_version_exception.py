from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data('2.35', '2.78')
def test_list_snapshots_api_version_exception(self, v):
    self.app.client_manager.share.api_version = api_versions.APIVersion(v)
    if v == '2.35':
        arglist = ['--description', 'Description']
        verifylist = [('description', 'Description')]
    elif v == '2.78':
        arglist = ['--count']
        verifylist = [('count', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)