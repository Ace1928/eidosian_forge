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
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume for which to modify type.')
@utils.arg('new_type', metavar='<volume-type>', help='New volume type.')
@utils.arg('--migration-policy', metavar='<never|on-demand>', required=False, choices=['never', 'on-demand'], default='never', help='Migration policy during retype of volume.')
def do_retype(cs, args):
    """Changes the volume type for a volume."""
    volume = utils.find_volume(cs, args.volume)
    volume.retype(args.new_type, args.migration_policy)