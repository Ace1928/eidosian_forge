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
@cliutils.arg('share', metavar='<share>', help='Name or ID of share to cancel migration.')
@api_versions.wraps('2.22')
def do_migration_cancel(cs, args):
    """Cancels migration of a given share when copying

    (Admin only, Experimental).
    """
    share = _find_share(cs, args.share)
    share.migration_cancel()