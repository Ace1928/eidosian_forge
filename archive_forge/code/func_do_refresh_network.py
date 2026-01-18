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
@utils.arg('server', metavar='<server>', help=_('Name or ID of a server for which the network cache should be refreshed from neutron (Admin only).'))
def do_refresh_network(cs, args):
    """Refresh server network information."""
    server = _find_server(cs, args.server)
    cs.server_external_events.create([{'server_uuid': server.id, 'name': 'network-changed'}])