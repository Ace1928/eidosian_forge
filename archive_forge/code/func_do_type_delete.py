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
@utils.arg('vol_type', metavar='<vol_type>', nargs='+', help='Name or ID of volume type or types to delete.')
def do_type_delete(cs, args):
    """Deletes volume type or types."""
    failure_count = 0
    for vol_type in args.vol_type:
        try:
            vtype = shell_utils.find_volume_type(cs, vol_type)
            cs.volume_types.delete(vtype)
            print('Request to delete volume type %s has been accepted.' % vol_type)
        except Exception as e:
            failure_count += 1
            print('Delete for volume type %s failed: %s' % (vol_type, e))
    if failure_count == len(args.vol_type):
        raise exceptions.CommandError('Unable to delete any of the specified types.')