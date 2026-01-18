import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('--all-tenants', dest='all_tenants', metavar='<0|1>', nargs='?', type=int, const=1, default=0, help='Shows details for all tenants. Admin only.')
@utils.arg('--status', metavar='<status>', default=None, help='Filters results by a status. Default=None.')
@utils.arg('--consistencygroup-id', metavar='<consistencygroup_id>', default=None, help='Filters results by a consistency group ID. Default=None.')
def do_cgsnapshot_list(cs, args):
    """Lists all cgsnapshots."""
    all_tenants = int(os.environ.get('ALL_TENANTS', args.all_tenants))
    search_opts = {'all_tenants': all_tenants, 'status': args.status, 'consistencygroup_id': args.consistencygroup_id}
    cgsnapshots = cs.cgsnapshots.list(search_opts=search_opts)
    columns = ['ID', 'Status', 'Name']
    shell_utils.print_list(cgsnapshots, columns)