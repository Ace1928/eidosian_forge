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
def do_aggregate_list(cs, args):
    """Print a list of all aggregates."""
    aggregates = cs.aggregates.list()
    columns = ['Id', 'Name', 'Availability Zone']
    if cs.api_version >= api_versions.APIVersion('2.41'):
        columns.append('UUID')
    utils.print_list(aggregates, columns)