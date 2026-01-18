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
@cliutils.arg('share_group', metavar='<share_group>', nargs='+', help='Name or ID of the share group(s).')
@cliutils.arg('--force', action='store_true', default=False, help='Attempt to force delete the share group (Default=False) (Admin only).')
@cliutils.arg('--wait', action='store_true', default=False, help='Wait for share group to delete')
@cliutils.service_type('sharev2')
def do_share_group_delete(cs, args):
    """Delete one or more share groups."""
    failure_count = 0
    share_group_to_delete = []
    for share_group in args.share_group:
        try:
            share_group_ref = _find_share_group(cs, share_group)
            share_group_to_delete.append(share_group_ref)
            share_group_ref.delete(args.force)
        except Exception as e:
            failure_count += 1
            print('Delete for share group %s failed: %s' % (share_group, e), file=sys.stderr)
    if failure_count == len(args.share_group):
        raise exceptions.CommandError('Unable to delete any of the specified share groups.')
    if args.wait:
        for share_group in share_group_to_delete:
            try:
                _wait_for_resource_status(cs, share_group, resource_type='share_group', expected_status='deleted')
            except exceptions.CommandError as e:
                print(e, file=sys.stderr)