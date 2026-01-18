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
@cliutils.arg('share_group_type', metavar='<share_group_type>', help='Name or ID of the share group type.')
@cliutils.arg('action', metavar='<action>', choices=['set', 'unset'], help="Actions: 'set' or 'unset'.")
@cliutils.arg('group_specs', metavar='<key=value>', nargs='*', default=None, help='Group specs to set or unset (only key is necessary to unset).')
@cliutils.service_type('sharev2')
def do_share_group_type_key(cs, args):
    """Set or unset group_spec for a share group type (Admin only)."""
    sg_type = _find_share_group_type(cs, args.share_group_type)
    if args.group_specs is not None:
        keypair = _extract_group_specs(args)
        if args.action == 'set':
            sg_type.set_keys(keypair)
        elif args.action == 'unset':
            sg_type.unset_keys(list(keypair))