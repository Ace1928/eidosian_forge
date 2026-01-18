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
@utils.arg('volume', metavar='<volume>', help='ID of the volume to update.')
@utils.arg('bootable', metavar='<True|true|False|false>', choices=['True', 'true', 'False', 'false'], help='Flag to indicate whether volume is bootable.')
def do_set_bootable(cs, args):
    """Update bootable status of a volume."""
    volume = utils.find_volume(cs, args.volume)
    cs.volumes.set_bootable(volume, strutils.bool_from_string(args.bootable, strict=True))