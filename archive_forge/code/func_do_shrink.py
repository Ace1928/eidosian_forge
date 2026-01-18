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
@cliutils.arg('share', metavar='<share>', help='Name or ID of share to shrink.')
@cliutils.arg('new_size', metavar='<new_size>', type=int, help='New size of share, in GiBs.')
@cliutils.arg('--wait', action='store_true', help='Wait for share shrinkage')
@cliutils.service_type('sharev2')
def do_shrink(cs, args):
    """Decreases the size of an existing share."""
    share = _find_share(cs, args.share)
    cs.shares.shrink(share, args.new_size)
    if args.wait:
        share = _wait_for_share_status(cs, share)
    else:
        share = _find_share(cs, args.share)
    _print_share(cs, share)