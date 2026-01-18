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
@cliutils.arg('share_group_snapshot', metavar='<share_group_snapshot>', nargs='+', help='Name or ID of the share group snapshot(s) to delete.')
@cliutils.arg('--force', action='store_true', default=False, help='Attempt to force delete the share group snapshot(s) (Default=False) (Admin only).')
@cliutils.service_type('sharev2')
def do_share_group_snapshot_delete(cs, args):
    """Remove one or more share group snapshots."""
    failure_count = 0
    kwargs = {}
    kwargs['force'] = args.force
    for sg_snapshot in args.share_group_snapshot:
        try:
            sg_snapshot_ref = _find_share_group_snapshot(cs, sg_snapshot)
            cs.share_group_snapshots.delete(sg_snapshot_ref, **kwargs)
        except Exception as e:
            failure_count += 1
            print('Delete for share group snapshot %s failed: %s' % (sg_snapshot, e), file=sys.stderr)
    if failure_count == len(args.share_group_snapshot):
        raise exceptions.CommandError('Unable to delete any of the specified share group snapshots.')