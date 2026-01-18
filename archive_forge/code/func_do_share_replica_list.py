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
@cliutils.arg('--share-id', '--share_id', '--si', metavar='<share_id>', default=None, action='single_alias', help='List replicas belonging to share.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "replica_state,id".')
@api_versions.wraps('2.11')
def do_share_replica_list(cs, args):
    """List share replicas."""
    share = _find_share(cs, args.share_id) if args.share_id else None
    if args.columns is not None:
        list_of_keys = _split_columns(columns=args.columns)
    else:
        list_of_keys = ['ID', 'Status', 'Replica State', 'Share ID', 'Host', 'Availability Zone', 'Updated At']
    if share:
        replicas = cs.share_replicas.list(share)
    else:
        replicas = cs.share_replicas.list()
    cliutils.print_list(replicas, list_of_keys)