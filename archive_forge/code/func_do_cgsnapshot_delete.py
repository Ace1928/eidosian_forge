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
@utils.arg('cgsnapshot', metavar='<cgsnapshot>', nargs='+', help='Name or ID of one or more cgsnapshots to be deleted.')
def do_cgsnapshot_delete(cs, args):
    """Removes one or more cgsnapshots."""
    failure_count = 0
    for cgsnapshot in args.cgsnapshot:
        try:
            shell_utils.find_cgsnapshot(cs, cgsnapshot).delete()
        except Exception as e:
            failure_count += 1
            print('Delete for cgsnapshot %s failed: %s' % (cgsnapshot, e))
    if failure_count == len(args.cgsnapshot):
        raise exceptions.CommandError('Unable to delete any of the specified cgsnapshots.')