import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_soft_deleted_api_version_exception(self):
    self.app.client_manager.share.api_version = api_versions.APIVersion('2.60')
    arglist = ['--soft-deleted']
    verifylist = [('soft_deleted', True)]
    search_opts = self._get_search_opts()
    search_opts['is_soft_deleted'] = True
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.assertRaises(osc_exceptions.CommandError, self.cmd.take_action, parsed_args)