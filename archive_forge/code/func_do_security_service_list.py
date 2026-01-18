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
@cliutils.arg('--all-tenants', '--all-projects', action='single_alias', dest='all_projects', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Display information from all projects (Admin only).')
@cliutils.arg('--share-network', '--share_network', metavar='<share_network>', action='single_alias', default=None, help='Filter results by share network id or name.')
@cliutils.arg('--status', metavar='<status>', default=None, help='Filter results by status.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Filter results by name.')
@cliutils.arg('--type', metavar='<type>', default=None, help='Filter results by type.')
@cliutils.arg('--user', metavar='<user>', default=None, help='Filter results by user or group used by projects.')
@cliutils.arg('--dns-ip', '--dns_ip', metavar='<dns_ip>', action='single_alias', default=None, help="Filter results by DNS IP address used inside project's network.")
@cliutils.arg('--ou', metavar='<ou>', default=None, help='Filter results by security service OU (Organizational Unit). Available only for microversion >= 2.44.')
@cliutils.arg('--server', metavar='<server>', default=None, help='Filter results by security service IP address or hostname.')
@cliutils.arg('--domain', metavar='<domain>', default=None, help='Filter results by domain.')
@cliutils.arg('--detailed', dest='detailed', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Show detailed information about filtered security services.')
@cliutils.arg('--offset', metavar='<offset>', default=None, help='Start position of security services listing.')
@cliutils.arg('--limit', metavar='<limit>', default=None, help='Number of security services to return per request.')
@cliutils.arg('--default-ad-site', metavar='<default_ad_site>', dest='default_ad_site', default=None, help='Default AD site. Available only for microversion >= 2.76.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "name,type".')
def do_security_service_list(cs, args):
    """Get a list of security services."""
    all_projects = int(os.environ.get('ALL_TENANTS', os.environ.get('ALL_PROJECTS', args.all_projects)))
    search_opts = {'all_tenants': all_projects, 'status': args.status, 'name': args.name, 'type': args.type, 'user': args.user, 'dns_ip': args.dns_ip, 'server': args.server, 'domain': args.domain, 'offset': args.offset, 'limit': args.limit}
    if cs.api_version.matches(api_versions.APIVersion('2.44'), api_versions.APIVersion()):
        search_opts['ou'] = args.ou
    elif args.ou:
        raise exceptions.CommandError('Security service Organizational Unit (ou) option is only available with manila API version >= 2.44')
    if cs.api_version.matches(api_versions.APIVersion('2.76'), api_versions.APIVersion()):
        search_opts['default_ad_site'] = args.default_ad_site
    elif args.ou:
        raise exceptions.CommandError('Security service Default AD site option is only available with manila API version >= 2.76')
    if args.share_network:
        search_opts['share_network_id'] = _find_share_network(cs, args.share_network).id
    security_services = cs.security_services.list(search_opts=search_opts, detailed=args.detailed)
    fields = ['id', 'name', 'status', 'type']
    if args.columns is not None:
        fields = _split_columns(columns=args.columns)
    if args.detailed:
        fields.append('share_networks')
    cliutils.print_list(security_services, fields=fields)