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
@cliutils.arg('share', metavar='<share>', nargs='+', help='Name or ID of the share(s).')
@cliutils.service_type('sharev2')
@api_versions.wraps('2.69')
def do_soft_delete(cs, args):
    """Soft delete one or more shares."""
    failure_count = 0
    for share in args.share:
        try:
            share_ref = _find_share(cs, share)
            cs.shares.soft_delete(share_ref)
        except Exception as e:
            failure_count += 1
            print('Soft deletion of share %s failed: %s' % (share, e), file=sys.stderr)
    if failure_count == len(args.share):
        raise exceptions.CommandError('Unable to soft delete any of the specified shares.')