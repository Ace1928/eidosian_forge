import argparse
import collections
import datetime
import getpass
import logging
import os
import pprint
import sys
import time
from oslo_utils import netutils
from oslo_utils import strutils
from oslo_utils import timeutils
import novaclient
from novaclient import api_versions
from novaclient import base
from novaclient import client
from novaclient import exceptions
from novaclient.i18n import _
from novaclient import shell
from novaclient import utils
from novaclient.v2 import availability_zones
from novaclient.v2 import quotas
from novaclient.v2 import servers
@utils.arg('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_("Maximum number of server groups to display. If limit is bigger than 'CONF.api.max_limit' option of Nova API, limit 'CONF.api.max_limit' will be used instead."))
@utils.arg('--offset', dest='offset', metavar='<offset>', type=int, default=None, help=_('The offset of groups list to display; use with limit to return a slice of server groups.'))
@utils.arg('--all-projects', dest='all_projects', action='store_true', default=False, help=_('Display server groups from all projects (Admin only).'))
def do_server_group_list(cs, args):
    """Print a list of all server groups."""
    server_groups = cs.server_groups.list(all_projects=args.all_projects, limit=args.limit, offset=args.offset)
    _print_server_group_details(cs, server_groups)