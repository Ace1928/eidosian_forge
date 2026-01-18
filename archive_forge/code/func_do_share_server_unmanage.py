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
@api_versions.wraps('2.49')
@cliutils.arg('share_server', metavar='<share_server>', nargs='+', help='ID of the share server(s).')
@cliutils.arg('--force', dest='force', action='store_true', required=False, default=False, help='Enforces the unmanage share server operation, even if the back-end driver does not support it.')
def do_share_server_unmanage(cs, args):
    """Unmanage share server (Admin only)."""
    failure_count = 0
    for server in args.share_server:
        try:
            cs.share_servers.unmanage(server, args.force)
        except Exception as e:
            failure_count += 1
            print('Unmanage for share server %s failed: %s' % (server, e), file=sys.stderr)
    if failure_count == len(args.share_server):
        raise exceptions.CommandError('Unable to unmanage any of the specified share servers.')