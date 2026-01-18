import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.68')
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume to reimage')
@utils.arg('image_id', metavar='<image-id>', help='The image id of the image that will be used to reimage the volume.')
@utils.arg('--reimage-reserved', metavar='<True|False>', default=False, help='Enables or disables reimage for a volume that is in reserved state otherwise only volumes in "available"  or "error" status may be re-imaged. Default=False.')
def do_reimage(cs, args):
    """Rebuilds a volume, overwriting all content with the specified image"""
    volume = utils.find_volume(cs, args.volume)
    volume.reimage(args.image_id, args.reimage_reserved)