import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.14')
@utils.arg('group_snapshot', metavar='<group_snapshot>', nargs='+', help='Name or ID of one or more group snapshots to be deleted.')
def do_group_snapshot_delete(cs, args):
    """Removes one or more group snapshots."""
    failure_count = 0
    for group_snapshot in args.group_snapshot:
        try:
            shell_utils.find_group_snapshot(cs, group_snapshot).delete()
        except Exception as e:
            failure_count += 1
            print('Delete for group snapshot %s failed: %s' % (group_snapshot, e))
    if failure_count == len(args.group_snapshot):
        raise exceptions.CommandError('Unable to delete any of the specified group snapshots.')