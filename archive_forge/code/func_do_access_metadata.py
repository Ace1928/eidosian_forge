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
@api_versions.wraps('2.45')
@cliutils.arg('access_id', metavar='<access_id>', help='ID of the NAS share access rule.')
@cliutils.arg('action', metavar='<action>', choices=['set', 'unset'], help="Actions: 'set' or 'unset'.")
@cliutils.arg('metadata', metavar='<key=value>', nargs='+', default=[], help='Space separated key=value pairs of metadata items to set. To unset only keys are required. ')
def do_access_metadata(cs, args):
    """Set or delete metadata on a share access rule."""
    share_access = cs.share_access_rules.get(args.access_id)
    metadata = _extract_metadata(args)
    if args.action == 'set':
        cs.share_access_rules.set_metadata(share_access, metadata)
    elif args.action == 'unset':
        cs.share_access_rules.unset_metadata(share_access, sorted(list(metadata), reverse=True))