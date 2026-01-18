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
@api_versions.wraps('2.29')
@cliutils.arg('share', metavar='<share>', help='Name or ID of share to migrate.')
@cliutils.arg('host', metavar='<host@backend#pool>', help="Destination host where share will be migrated to. Use the format 'host@backend#pool'.")
@cliutils.arg('--force_host_assisted_migration', '--force-host-assisted-migration', metavar='<True|False>', choices=['True', 'False'], action='single_alias', required=False, default=False, help='Enforces the use of the host-assisted migration approach, which bypasses driver optimizations. Default=False.')
@cliutils.arg('--preserve-metadata', '--preserve_metadata', action='single_alias', metavar='<True|False>', choices=['True', 'False'], required=True, help='Enforces migration to preserve all file metadata when moving its contents. If set to True, host-assisted migration will not be attempted.')
@cliutils.arg('--preserve-snapshots', '--preserve_snapshots', action='single_alias', metavar='<True|False>', choices=['True', 'False'], required=True, help='Enforces migration of the share snapshots to the destination. If set to True, host-assisted migration will not be attempted.')
@cliutils.arg('--writable', metavar='<True|False>', choices=['True', 'False'], required=True, help='Enforces migration to keep the share writable while contents are being moved. If set to True, host-assisted migration will not be attempted.')
@cliutils.arg('--nondisruptive', metavar='<True|False>', choices=['True', 'False'], required=True, help='Enforces migration to be nondisruptive. If set to True, host-assisted migration will not be attempted.')
@cliutils.arg('--new_share_network', '--new-share-network', metavar='<new_share_network>', action='single_alias', required=False, help='Specify the new share network for the share. Do not specify this parameter if the migrating share has to be retained within its current share network.', default=None)
@cliutils.arg('--new_share_type', '--new-share-type', metavar='<new_share_type>', required=False, action='single_alias', help='Specify the new share type for the share. Do not specify this parameter if the migrating share has to be retained with its current share type.', default=None)
def do_migration_start(cs, args):
    """Migrates share to a new host (Admin only, Experimental)."""
    share = _find_share(cs, args.share)
    new_share_net_id = None
    if args.new_share_network:
        share_net = _find_share_network(cs, args.new_share_network)
        new_share_net_id = share_net.id if share_net else None
    new_share_type_id = None
    if args.new_share_type:
        share_type = _find_share_type(cs, args.new_share_type)
        new_share_type_id = share_type.id if share_type else None
    share.migration_start(args.host, args.force_host_assisted_migration, args.preserve_metadata, args.writable, args.nondisruptive, args.preserve_snapshots, new_share_net_id, new_share_type_id)