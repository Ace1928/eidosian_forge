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
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume to transfer.')
@utils.arg('--name', metavar='<name>', default=None, help='Transfer name. Default=None.')
@utils.arg('--display-name', help=argparse.SUPPRESS)
def do_transfer_create(cs, args):
    """Creates a volume transfer."""
    if args.display_name is not None:
        args.name = args.display_name
    volume = utils.find_volume(cs, args.volume)
    transfer = cs.transfers.create(volume.id, args.name)
    info = dict()
    info.update(transfer._info)
    info.pop('links', None)
    shell_utils.print_dict(info)