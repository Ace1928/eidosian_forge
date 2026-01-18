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
@utils.arg('name', metavar='<name>', help='Name of new volume type.')
@utils.arg('--description', metavar='<description>', help='Description of new volume type.')
@utils.arg('--is-public', metavar='<is-public>', default=True, help='Make type accessible to the public (default true).')
def do_type_create(cs, args):
    """Creates a volume type."""
    is_public = strutils.bool_from_string(args.is_public, strict=True)
    vtype = cs.volume_types.create(args.name, args.description, is_public)
    shell_utils.print_volume_type_list([vtype])