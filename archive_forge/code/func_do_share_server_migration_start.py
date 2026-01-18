from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('share_server_id', metavar='<share_server_id>', help='ID of the share server to migrate.')
@cliutils.arg('host', metavar='<host@backend>', help="Destination to migrate the share server to. Use the format '<node_hostname>@<backend_name>'.")
@cliutils.arg('--preserve-snapshots', '--preserve_snapshots', action='single_alias', metavar='<True|False>', choices=['True', 'False'], required=True, help='Set to True if snapshots must be preserved at the migration destination.')
@cliutils.arg('--writable', metavar='<True|False>', choices=['True', 'False'], required=True, help='Enforces migration to keep all its shares writable  while contents are being moved.')
@cliutils.arg('--nondisruptive', metavar='<True|False>', choices=['True', 'False'], required=True, help='Enforces migration to be nondisruptive.')
@cliutils.arg('--new_share_network', '--new-share-network', metavar='<new_share_network>', action='single_alias', required=False, help='Specify a new share network for the share server. Do not specify this parameter if the migrating share server has to be retained within its current share network.', default=None)
@api_versions.wraps('2.57')
@api_versions.experimental_api
def do_share_server_migration_start(cs, args):
    """Migrates share server to a new host (Admin only, Experimental)."""
    share_server = _find_share_server(cs, args.share_server_id)
    new_share_net_id = None
    if args.new_share_network:
        share_net = _find_share_network(cs, args.new_share_network)
        new_share_net_id = share_net.id
    share_server.migration_start(args.host, args.writable, args.nondisruptive, args.preserve_snapshots, new_share_net_id)