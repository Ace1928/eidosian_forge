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
@utils.arg('aggregate', metavar='<aggregate>', help=_('Name or ID of aggregate to delete.'))
def do_aggregate_delete(cs, args):
    """Delete the aggregate."""
    aggregate = _find_aggregate(cs, args.aggregate)
    cs.aggregates.delete(aggregate)
    print(_('Aggregate %s has been successfully deleted.') % aggregate.id)