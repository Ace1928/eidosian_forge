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
@utils.arg('volume', metavar='<volume>', help='ID of volume.')
def do_image_metadata_show(cs, args):
    """Shows volume image metadata."""
    volume = utils.find_volume(cs, args.volume)
    resp, body = volume.show_image_metadata(volume)
    shell_utils.print_dict(body['metadata'], 'Metadata-property')