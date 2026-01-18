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
@cliutils.arg('id', metavar='<id>', nargs='+', type=str, help='ID of the share server(s) to delete.')
@cliutils.arg('--wait', action='store_true', help='Wait for share server to delete')
@cliutils.service_type('sharev2')
def do_share_server_delete(cs, args):
    """Delete one or more share servers (Admin only)."""
    failure_count = 0
    share_servers_to_delete = []
    for server_id in args.id:
        try:
            id_ref = _find_share_server(cs, server_id)
            share_servers_to_delete.append(id_ref)
            id_ref.delete()
        except Exception as e:
            failure_count += 1
            print('Delete for share server %s failed: %s' % (server_id, e), file=sys.stderr)
    if failure_count == len(args.id):
        raise exceptions.CommandError('Unable to delete any of the specified share servers.')
    if args.wait:
        for share_server in share_servers_to_delete:
            try:
                _wait_for_resource_status(cs, share_server, resource_type='share_server', expected_status='deleted')
            except exceptions.CommandError as e:
                print(e, file=sys.stderr)