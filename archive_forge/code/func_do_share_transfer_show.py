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
@cliutils.arg('transfer', metavar='<transfer>', help='Name or ID of transfer to show.')
def do_share_transfer_show(cs, args):
    """Delete a transfer."""
    transfer = _find_share_transfer(cs, args.transfer)
    _print_share_transfer(transfer)