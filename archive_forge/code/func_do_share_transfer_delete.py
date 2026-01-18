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
@cliutils.arg('transfer', metavar='<transfer>', nargs='+', help='ID or name of the transfer(s).')
def do_share_transfer_delete(cs, args):
    """Remove one or more transfers."""
    failure_count = 0
    for transfer in args.transfer:
        try:
            transfer_ref = _find_share_transfer(cs, transfer)
            transfer_ref.delete()
        except Exception as e:
            failure_count += 1
            print('Delete for share transfer %s failed: %s' % (transfer, e), file=sys.stderr)
    if failure_count == len(args.transfer):
        raise exceptions.CommandError('Unable to delete any of the specified transfers.')