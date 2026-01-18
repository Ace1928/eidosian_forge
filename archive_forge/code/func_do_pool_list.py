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
@cliutils.arg('--host', metavar='<host>', type=str, default='.*', help='Filter results by host name.  Regular expressions are supported.')
@cliutils.arg('--backend', metavar='<backend>', type=str, default='.*', help='Filter results by backend name.  Regular expressions are supported.')
@cliutils.arg('--pool', metavar='<pool>', type=str, default='.*', help='Filter results by pool name.  Regular expressions are supported.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "name,host".')
@cliutils.arg('--detail', '--detailed', action='store_true', help='Show detailed information about pools. If this parameter is set to True, --columns parameter will be ignored if present. (Default=False)')
@cliutils.arg('--share-type', '--share_type', '--share-type-id', '--share_type_id', metavar='<share_type>', type=str, default=None, action='single_alias', help='Filter results by share type name or ID. (Default=None)Available only for microversion >= 2.23.')
def do_pool_list(cs, args):
    """List all backend storage pools known to the scheduler (Admin only)."""
    search_opts = {'host': args.host, 'backend': args.backend, 'pool': args.pool, 'share_type': args.share_type}
    if args.detail:
        fields = ['Name', 'Host', 'Backend', 'Pool', 'Capabilities']
    else:
        fields = ['Name', 'Host', 'Backend', 'Pool']
    pools = cs.pools.list(detailed=args.detail, search_opts=search_opts)
    if args.columns is not None:
        fields = _split_columns(columns=args.columns)
        pools = cs.pools.list(detailed=True, search_opts=search_opts)
    if args.detail:
        for info in pools:
            backend = dict()
            backend['name'] = info.name
            backend.update(info.capabilities)
            cliutils.print_dict(backend)
    else:
        cliutils.print_list(pools, fields=fields)