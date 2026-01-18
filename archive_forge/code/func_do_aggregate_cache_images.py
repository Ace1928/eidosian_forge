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
@api_versions.wraps('2.81')
@utils.arg('aggregate', metavar='<aggregate>', help=_('Name or ID of the aggregate.'))
@utils.arg('images', metavar='<image>', nargs='+', help=_('Name or ID of image(s) to cache on the hosts within the aggregate.'))
def do_aggregate_cache_images(cs, args):
    """Request images be cached."""
    aggregate = _find_aggregate(cs, args.aggregate)
    images = _find_images(cs, args.images)
    cs.aggregates.cache_images(aggregate, images)