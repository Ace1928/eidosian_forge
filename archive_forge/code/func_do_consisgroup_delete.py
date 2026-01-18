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
@utils.arg('consistencygroup', metavar='<consistencygroup>', nargs='+', help='Name or ID of one or more consistency groups to be deleted.')
@utils.arg('--force', action='store_true', default=False, help='Allows or disallows consistency groups to be deleted. If the consistency group is empty, it can be deleted without the force flag. If the consistency group is not empty, the force flag is required for it to be deleted.')
def do_consisgroup_delete(cs, args):
    """Removes one or more consistency groups."""
    failure_count = 0
    for consistencygroup in args.consistencygroup:
        try:
            shell_utils.find_consistencygroup(cs, consistencygroup).delete(args.force)
        except Exception as e:
            failure_count += 1
            print('Delete for consistency group %s failed: %s' % (consistencygroup, e))
    if failure_count == len(args.consistencygroup):
        raise exceptions.CommandError('Unable to delete any of the specified consistency groups.')