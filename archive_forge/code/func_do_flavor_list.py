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
@utils.arg('--extra-specs', dest='extra_specs', action='store_true', default=False, help=_('Get extra-specs of each flavor.'))
@utils.arg('--all', dest='all', action='store_true', default=False, help=_('Display all flavors (Admin only).'))
@utils.arg('--marker', dest='marker', metavar='<marker>', default=None, help=_('The last flavor ID of the previous page; displays list of flavors after "marker".'))
@utils.arg('--min-disk', dest='min_disk', metavar='<min-disk>', default=None, help=_('Filters the flavors by a minimum disk space, in GiB.'))
@utils.arg('--min-ram', dest='min_ram', metavar='<min-ram>', default=None, help=_('Filters the flavors by a minimum RAM, in MiB.'))
@utils.arg('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_("Maximum number of flavors to display. If limit is bigger than 'CONF.api.max_limit' option of Nova API, limit 'CONF.api.max_limit' will be used instead."))
@utils.arg('--sort-key', dest='sort_key', metavar='<sort-key>', default=None, help=_('Flavors list sort key.'))
@utils.arg('--sort-dir', dest='sort_dir', metavar='<sort-dir>', default=None, help=_('Flavors list sort direction.'))
def do_flavor_list(cs, args):
    """Print a list of available 'flavors' (sizes of servers)."""
    if args.all:
        flavors = cs.flavors.list(is_public=None, min_disk=args.min_disk, min_ram=args.min_ram, sort_key=args.sort_key, sort_dir=args.sort_dir)
    else:
        flavors = cs.flavors.list(marker=args.marker, min_disk=args.min_disk, min_ram=args.min_ram, sort_key=args.sort_key, sort_dir=args.sort_dir, limit=args.limit)
    _print_flavor_list(cs, flavors, args.extra_specs)