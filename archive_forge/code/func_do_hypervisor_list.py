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
@api_versions.wraps('2.33')
@utils.arg('--matching', metavar='<hostname>', default=None, help=_('List hypervisors matching the given <hostname> (or pattern). If matching is used limit and marker options will be ignored.'))
@utils.arg('--marker', dest='marker', metavar='<marker>', default=None, help=_('The last hypervisor of the previous page; displays list of hypervisors after "marker".'))
@utils.arg('--limit', dest='limit', metavar='<limit>', type=int, default=None, help=_("Maximum number of hypervisors to display. If limit is bigger than 'CONF.api.max_limit' option of Nova API, limit 'CONF.api.max_limit' will be used instead."))
def do_hypervisor_list(cs, args):
    """List hypervisors."""
    _do_hypervisor_list(cs, matching=args.matching, limit=args.limit, marker=args.marker)