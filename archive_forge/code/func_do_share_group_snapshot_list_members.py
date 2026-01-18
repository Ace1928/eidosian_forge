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
@cliutils.arg('share_group_snapshot', metavar='<share_group_snapshot>', help='Name or ID of the share group snapshot.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,name".')
@cliutils.service_type('sharev2')
def do_share_group_snapshot_list_members(cs, args):
    """List members of a share group snapshot."""
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ('Share ID', 'Size')
    sg_snapshot = _find_share_group_snapshot(cs, args.share_group_snapshot)
    members = [type('ShareGroupSnapshotMember', (object,), member) for member in sg_snapshot._info.get('members', [])]
    cliutils.print_list(members, fields=list_of_keys)