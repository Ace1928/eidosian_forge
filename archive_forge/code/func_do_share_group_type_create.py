from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('name', metavar='<name>', help='Name of the new share group type.')
@cliutils.arg('share_types', metavar='<share_types>', type=str, help='Comma-separated list of share type names or IDs.')
@cliutils.arg('--is_public', '--is-public', metavar='<is_public>', action='single_alias', help='Make type accessible to the public (default true).')
@cliutils.arg('--group-specs', '--group_specs', metavar='<key=value>', type=str, nargs='*', action='single_alias', default=None, help='Share Group type extra specs by key and value. OPTIONAL: Default=None. Example: "--group-specs consistent_snapshot_support=host".')
@cliutils.service_type('sharev2')
def do_share_group_type_create(cs, args):
    """Create a new share group type (Admin only)."""
    share_types = [_find_share_type(cs, share_type) for share_type in args.share_types.split(',')]
    kwargs = {'share_types': share_types, 'name': args.name, 'is_public': strutils.bool_from_string(args.is_public, default=True)}
    if args.group_specs is not None:
        kwargs['group_specs'] = _extract_group_specs(args)
    sg_type = cs.share_group_types.create(**kwargs)
    _print_share_group_type(sg_type)