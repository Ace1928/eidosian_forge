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
@utils.arg('snapshot', metavar='<snapshot>', help='ID of snapshot for which to update metadata.')
@utils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='Metadata key and value pair to update.')
def do_snapshot_metadata_update_all(cs, args):
    """Updates snapshot metadata."""
    snapshot = shell_utils.find_volume_snapshot(cs, args.snapshot)
    metadata = shell_utils.extract_metadata(args)
    metadata = snapshot.update_all_metadata(metadata)
    shell_utils.print_dict(metadata)