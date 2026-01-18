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
@api_versions.wraps('3.13')
@utils.arg('group', metavar='<group>', nargs='+', help='Name or ID of one or more groups to be deleted.')
@utils.arg('--delete-volumes', action='store_true', default=False, help='Allows or disallows groups to be deleted if they are not empty. If the group is empty, it can be deleted without the delete-volumes flag. If the group is not empty, the delete-volumes flag is required for it to be deleted. If True, all volumes in the group will also be deleted.')
def do_group_delete(cs, args):
    """Removes one or more groups."""
    failure_count = 0
    for group in args.group:
        try:
            shell_utils.find_group(cs, group).delete(args.delete_volumes)
        except Exception as e:
            failure_count += 1
            print('Delete for group %s failed: %s' % (group, e))
    if failure_count == len(args.group):
        raise exceptions.CommandError('Unable to delete any of the specified groups.')