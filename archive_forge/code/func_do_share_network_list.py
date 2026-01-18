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
@api_versions.wraps('2.26')
@cliutils.arg('--all-tenants', '--all-projects', action='single_alias', dest='all_projects', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Display information from all projects (Admin only).')
@cliutils.arg('--project-id', '--project_id', metavar='<project_id>', action='single_alias', default=None, help='Filter results by project ID.')
@cliutils.arg('--name', metavar='<name>', type=str, default=None, help='Filter results by name.')
@cliutils.arg('--description', metavar='<description>', type=str, default=None, help='Filter results by description. Available only for microversion >= 2.36.')
@cliutils.arg('--created-since', '--created_since', metavar='<created_since>', action='single_alias', default=None, help="Return only share networks created since given date. The date is in the format 'yyyy-mm-dd'.")
@cliutils.arg('--created-before', '--created_before', metavar='<created_before>', action='single_alias', default=None, help="Return only share networks created until given date. The date is in the format 'yyyy-mm-dd'.")
@cliutils.arg('--security-service', '--security_service', metavar='<security_service>', action='single_alias', default=None, help='Filter results by attached security service.')
@cliutils.arg('--neutron-net-id', '--neutron_net_id', '--neutron_net-id', '--neutron-net_id', metavar='<neutron_net_id>', action='single_alias', default=None, help='Filter results by neutron net ID.')
@cliutils.arg('--neutron-subnet-id', '--neutron_subnet_id', '--neutron-subnet_id', '--neutron_subnet-id', metavar='<neutron_subnet_id>', action='single_alias', default=None, help='Filter results by neutron subnet ID.')
@cliutils.arg('--network-type', '--network_type', metavar='<network_type>', action='single_alias', default=None, help='Filter results by network type.')
@cliutils.arg('--segmentation-id', '--segmentation_id', metavar='<segmentation_id>', type=int, action='single_alias', default=None, help='Filter results by segmentation ID.')
@cliutils.arg('--cidr', metavar='<cidr>', default=None, help='Filter results by CIDR.')
@cliutils.arg('--ip-version', '--ip_version', metavar='<ip_version>', type=int, action='single_alias', default=None, help='Filter results by IP version.')
@cliutils.arg('--offset', metavar='<offset>', type=int, default=None, help='Start position of share networks listing.')
@cliutils.arg('--limit', metavar='<limit>', type=int, default=None, help='Number of share networks to return per request.')
@cliutils.arg('--columns', metavar='<columns>', type=str, default=None, help='Comma separated list of columns to be displayed example --columns "id".')
@cliutils.arg('--name~', metavar='<name~>', type=str, default=None, help='Filter results matching a share network name pattern. Available only for microversion >= 2.36.')
@cliutils.arg('--description~', metavar='<description~>', type=str, default=None, help='Filter results matching a share network description pattern. Available only for microversion >= 2.36.')
def do_share_network_list(cs, args):
    """Get a list of share networks"""
    all_projects = int(os.environ.get('ALL_TENANTS', os.environ.get('ALL_PROJECTS', args.all_projects)))
    search_opts = {'all_tenants': all_projects, 'project_id': args.project_id, 'name': args.name, 'created_since': args.created_since, 'created_before': args.created_before, 'neutron_net_id': args.neutron_net_id, 'neutron_subnet_id': args.neutron_subnet_id, 'network_type': args.network_type, 'segmentation_id': args.segmentation_id, 'cidr': args.cidr, 'ip_version': args.ip_version, 'offset': args.offset, 'limit': args.limit}
    if cs.api_version.matches(api_versions.APIVersion('2.36'), api_versions.APIVersion()):
        search_opts['name~'] = getattr(args, 'name~')
        search_opts['description~'] = getattr(args, 'description~')
        search_opts['description'] = getattr(args, 'description')
    elif getattr(args, 'name~') or getattr(args, 'description~') or getattr(args, 'description'):
        raise exceptions.CommandError('Pattern based filtering (name~, description~ and description) is only available with manila API version >= 2.36')
    if args.security_service:
        search_opts['security_service_id'] = _find_security_service(cs, args.security_service).id
    share_networks = cs.share_networks.list(search_opts=search_opts)
    fields = ['id', 'name']
    if args.columns is not None:
        fields = _split_columns(columns=args.columns)
    cliutils.print_list(share_networks, fields=fields)