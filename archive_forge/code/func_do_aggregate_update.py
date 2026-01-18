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
@utils.arg('aggregate', metavar='<aggregate>', help=_('Name or ID of aggregate to update.'))
@utils.arg('--name', metavar='<name>', dest='name', help=_('New name for aggregate.'))
@utils.arg('--availability-zone', metavar='<availability-zone>', dest='availability_zone', help=_('New availability zone for aggregate.'))
def do_aggregate_update(cs, args):
    """Update the aggregate's name and optionally availability zone."""
    aggregate = _find_aggregate(cs, args.aggregate)
    updates = {}
    if args.name:
        updates['name'] = args.name
    if args.availability_zone:
        updates['availability_zone'] = args.availability_zone
    if not updates:
        raise exceptions.CommandError(_("Either '--name <name>' or '--availability-zone <availability-zone>' must be specified."))
    aggregate = cs.aggregates.update(aggregate.id, updates)
    print(_('Aggregate %s has been successfully updated.') % aggregate.id)
    _print_aggregate_details(cs, aggregate)