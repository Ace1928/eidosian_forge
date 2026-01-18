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
@utils.arg('--tenant', dest='tenant', metavar='<tenant>', nargs='?', help=_('Display information from single tenant (Admin only).'))
@utils.arg('--reserved', dest='reserved', action='store_true', default=False, help=_('Include reservations count.'))
def do_limits(cs, args):
    """Print rate and absolute limits."""
    limits = cs.limits.get(args.reserved, args.tenant)
    _print_rate_limits(limits.rate)
    _print_absolute_limits(limits.absolute)