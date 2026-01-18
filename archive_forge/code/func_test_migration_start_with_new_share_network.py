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
def test_migration_start_with_new_share_network(self):
    """Test with new_share_network"""
    arglist = [self._share.id, 'host@driver#pool', '--preserve-metadata', 'False', '--preserve-snapshots', 'False', '--writable', 'False', '--nondisruptive', 'False', '--new-share-network', self.new_share_network.id, '--force-host-assisted-migration', 'False']
    verifylist = [('share', self._share.id), ('host', 'host@driver#pool'), ('preserve_metadata', 'False'), ('preserve_snapshots', 'False'), ('writable', 'False'), ('nondisruptive', 'False'), ('new_share_network', self.new_share_network.id), ('force_host_assisted_migration', 'False')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self._share.migration_start.assert_called_with('host@driver#pool', 'False', 'False', 'False', 'False', 'False', self.new_share_network.id, None)
    self.assertIsNone(result)