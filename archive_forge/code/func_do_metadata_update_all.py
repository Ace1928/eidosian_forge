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
@utils.arg('volume', metavar='<volume>', help='ID of volume for which to update metadata.')
@utils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='Metadata key and value pair or pairs to update.')
def do_metadata_update_all(cs, args):
    """Updates volume metadata."""
    volume = utils.find_volume(cs, args.volume)
    metadata = shell_utils.extract_metadata(args)
    metadata = volume.update_all_metadata(metadata)
    shell_utils.print_dict(metadata['metadata'], 'Metadata-property')