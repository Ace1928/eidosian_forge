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
@utils.arg('volume', metavar='<volume>', nargs='+', help='Name or ID of volume or volumes to delete.')
def do_force_delete(cs, args):
    """Attempts force-delete of volume, regardless of state."""
    failure_count = 0
    for volume in args.volume:
        try:
            utils.find_volume(cs, volume).force_delete()
        except Exception as e:
            failure_count += 1
            print('Delete for volume %s failed: %s' % (volume, e))
    if failure_count == len(args.volume):
        raise exceptions.CommandError('Unable to force delete any of the specified volumes.')