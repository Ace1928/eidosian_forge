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
@cliutils.arg('--host', metavar='<hostname>', default=None, help='Filter results by name of host.')
@cliutils.arg('--status', metavar='<status>', default=None, help='Filter results by status.')
@cliutils.arg('--share-network', metavar='<share_network>', default=None, help='Filter results by share network.')
@cliutils.arg('--project-id', metavar='<project_id>', default=None, help='Filter results by project ID.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id,host,status".')
@cliutils.arg('--share-network-subnet', '--share_network_subnet', type=str, metavar='<share_network_subnet>', help="Filter results by share network subnet that the share server's network allocation exists whithin. Available for micro version >= 2.51 (Optional, Default=None).", default=None)
def do_share_server_list(cs, args):
    """List all share servers (Admin only)."""
    search_opts = {'host': args.host, 'share_network': args.share_network, 'status': args.status, 'project_id': args.project_id}
    fields = ['Id', 'Host', 'Status', 'Share Network', 'Project Id', 'Updated_at']
    if cs.api_version < api_versions.APIVersion('2.51'):
        if getattr(args, 'share_network_subnet'):
            raise exceptions.CommandError('Share network subnet option is only available with manila API version >= 2.51')
    elif cs.api_version < api_versions.APIVersion('2.70'):
        search_opts.update({'share_network_subnet_id': args.share_network_subnet})
        fields.append('Share Network Subnet Id')
    else:
        search_opts.update({'share_network_subnet_id': args.share_network_subnet})
        fields.append('Share Network Subnet IDs')
    if args.columns is not None:
        fields = _split_columns(columns=args.columns)
    share_servers = cs.share_servers.list(search_opts=search_opts)
    cliutils.print_list(share_servers, fields=fields)