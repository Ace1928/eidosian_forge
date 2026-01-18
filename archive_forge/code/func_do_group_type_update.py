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
@api_versions.wraps('3.11')
@utils.arg('id', metavar='<id>', help='ID of the group type.')
@utils.arg('--name', metavar='<name>', help='Name of the group type.')
@utils.arg('--description', metavar='<description>', help='Description of the group type.')
@utils.arg('--is-public', metavar='<is-public>', help='Make type accessible to the public or not.')
def do_group_type_update(cs, args):
    """Updates group type name, description, and/or is_public."""
    is_public = strutils.bool_from_string(args.is_public)
    gtype = cs.group_types.update(args.id, args.name, args.description, is_public)
    shell_utils.print_group_type_list([gtype])