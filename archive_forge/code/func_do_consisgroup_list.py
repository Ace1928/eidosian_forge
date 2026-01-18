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
def do_consisgroup_list(cs, args):
    """Lists all consistency groups."""
    search_opts = {'all_tenants': args.all_tenants}
    consistencygroups = cs.consistencygroups.list(search_opts=search_opts)
    columns = ['ID', 'Status', 'Name']
    shell_utils.print_list(consistencygroups, columns)