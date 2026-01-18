import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('snapshot', metavar='<snapshot>', nargs='+', help='Name or ID of the snapshot(s) to delete.')
@utils.arg('--force', action='store_true', help='Allows deleting snapshot of a volume when its status is other than "available" or "error". Default=False.')
def do_snapshot_delete(cs, args):
    """Removes one or more snapshots."""
    failure_count = 0
    for snapshot in args.snapshot:
        try:
            shell_utils.find_volume_snapshot(cs, snapshot).delete(args.force)
        except Exception as e:
            failure_count += 1
            print('Delete for snapshot %s failed: %s' % (snapshot, e))
    if failure_count == len(args.snapshot):
        raise exceptions.CommandError('Unable to delete any of the specified snapshots.')