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
@api_versions.wraps('2.77')
@cliutils.arg('share', metavar='<share>', help='Name or ID of share to transfer.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Transfer name. Default=None.')
def do_share_transfer_create(cs, args):
    """Creates a share transfer."""
    share = _find_share(cs, args.share)
    transfer = cs.transfers.create(share.id, args.name)
    _print_share_transfer(transfer)