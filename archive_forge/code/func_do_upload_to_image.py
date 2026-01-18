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
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume to snapshot.')
@utils.arg('--force', metavar='<True|False>', const=True, nargs='?', default=False, help='Enables or disables upload of a volume that is attached to an instance. Default=False. This option may not be supported by your cloud.')
@utils.arg('--container-format', metavar='<container-format>', default='bare', help='Container format type. Default is bare.')
@utils.arg('--container_format', help=argparse.SUPPRESS)
@utils.arg('--disk-format', metavar='<disk-format>', default='raw', help='Disk format type. Default is raw.')
@utils.arg('--disk_format', help=argparse.SUPPRESS)
@utils.arg('image_name', metavar='<image-name>', help='The new image name.')
@utils.arg('--image_name', help=argparse.SUPPRESS)
def do_upload_to_image(cs, args):
    """Uploads volume to Image Service as an image."""
    volume = utils.find_volume(cs, args.volume)
    shell_utils.print_volume_image(volume.upload_to_image(args.force, args.image_name, args.container_format, args.disk_format))