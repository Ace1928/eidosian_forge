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
@utils.arg('aggregate', metavar='<aggregate>', help=_('Name or ID of aggregate.'))
@utils.arg('host', metavar='<host>', help=_('The host to add to the aggregate.'))
def do_aggregate_add_host(cs, args):
    """Add the host to the specified aggregate."""
    aggregate = _find_aggregate(cs, args.aggregate)
    aggregate = cs.aggregates.add_host(aggregate.id, args.host)
    print(_('Host %(host)s has been successfully added for aggregate %(aggregate_id)s ') % {'host': args.host, 'aggregate_id': aggregate.id})
    _print_aggregate_details(cs, aggregate)