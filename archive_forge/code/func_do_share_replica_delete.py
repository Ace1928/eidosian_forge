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
@cliutils.arg('replica', metavar='<replica>', nargs='+', help='ID of the share replica.')
@cliutils.arg('--force', action='store_true', default=False, help='Attempt to force deletion of a replica on its backend. Using this option will purge the replica from Manila even if it is not cleaned up on the backend. Defaults to False.')
@api_versions.wraps('2.11')
def do_share_replica_delete(cs, args):
    """Remove one or more share replicas."""
    failure_count = 0
    kwargs = {'force': args.force}
    for replica in args.replica:
        try:
            replica_ref = _find_share_replica(cs, replica)
            cs.share_replicas.delete(replica_ref, **kwargs)
        except Exception as e:
            failure_count += 1
            print('Delete for share replica %s failed: %s' % (replica, e), file=sys.stderr)
    if failure_count == len(args.replica):
        raise exceptions.CommandError('Unable to delete any of the specified replicas.')