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
@utils.arg('snapshot', metavar='<snapshot>', help='ID of snapshot.')
def do_snapshot_metadata_show(cs, args):
    """Shows snapshot metadata."""
    snapshot = shell_utils.find_volume_snapshot(cs, args.snapshot)
    shell_utils.print_dict(snapshot._info['metadata'], 'Metadata-property')