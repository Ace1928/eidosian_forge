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
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume for which to update metadata.')
@utils.arg('action', metavar='<action>', choices=['set', 'unset'], help="The action. Valid values are 'set' or 'unset.'")
@utils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='Metadata key and value pair to set or unset. For unset, specify only the key.')
def do_image_metadata(cs, args):
    """Sets or deletes volume image metadata."""
    volume = utils.find_volume(cs, args.volume)
    metadata = shell_utils.extract_metadata(args)
    if args.action == 'set':
        cs.volumes.set_image_metadata(volume, metadata)
    elif args.action == 'unset':
        cs.volumes.delete_image_metadata(volume, sorted(metadata.keys(), reverse=True))