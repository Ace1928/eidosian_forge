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
@utils.arg('id', metavar='<id>', help='ID of the volume type.')
@utils.arg('--name', metavar='<name>', help='Name of the volume type.')
@utils.arg('--description', metavar='<description>', help='Description of the volume type.')
@utils.arg('--is-public', metavar='<is-public>', help='Make type accessible to the public or not.')
def do_type_update(cs, args):
    """Updates volume type name, description, and/or is_public."""
    is_public = args.is_public
    if args.name is None and args.description is None and (is_public is None):
        raise exceptions.CommandError('Specify a new type name, description, is_public or a combination thereof.')
    if is_public is not None:
        is_public = strutils.bool_from_string(args.is_public, strict=True)
    vtype = cs.volume_types.update(args.id, args.name, args.description, is_public)
    shell_utils.print_volume_type_list([vtype])