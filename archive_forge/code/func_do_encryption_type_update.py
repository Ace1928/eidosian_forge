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
@utils.arg('volume_type', metavar='<volume-type>', type=str, help='Name or ID of the volume type')
@utils.arg('--provider', metavar='<provider>', type=str, required=False, default=argparse.SUPPRESS, help="Encryption provider format (e.g. 'luks' or 'plain').")
@utils.arg('--cipher', metavar='<cipher>', type=str, nargs='?', required=False, default=argparse.SUPPRESS, const=None, help='Encryption algorithm/mode to use (e.g., aes-xts-plain64). Provide parameter without value to set to provider default.')
@utils.arg('--key-size', dest='key_size', metavar='<key-size>', type=int, nargs='?', required=False, default=argparse.SUPPRESS, const=None, help='Size of the encryption key, in bits (e.g., 128, 256). Provide parameter without value to set to provider default. ')
@utils.arg('--control-location', dest='control_location', metavar='<control-location>', choices=['front-end', 'back-end'], type=str, required=False, default=argparse.SUPPRESS, help="Notional service where encryption is performed (e.g., front-end=Nova). Values: 'front-end', 'back-end'")
def do_encryption_type_update(cs, args):
    """Update encryption type information for a volume type (Admin Only)."""
    volume_type = shell_utils.find_volume_type(cs, args.volume_type)
    body = {}
    for attr in ['provider', 'cipher', 'key_size', 'control_location']:
        if hasattr(args, attr):
            body[attr] = getattr(args, attr)
    cs.volume_encryption_types.update(volume_type, body)
    result = cs.volume_encryption_types.get(volume_type)
    shell_utils.print_volume_encryption_type_list([result])