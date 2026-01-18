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
@utils.arg('host', metavar='<host>', help='The hypervisor hostname (or pattern) to search for. WARNING: Use a fully qualified domain name if you only want to cold migrate from a specific host.')
@utils.arg('--strict', dest='strict', action='store_true', default=False, help=_('Migrate host with exact hypervisor hostname match'))
def do_host_servers_migrate(cs, args):
    """Cold migrate all instances off the specified host to other available
    hosts.
    """
    response = []
    for server in _hyper_servers(cs, args.host, args.strict):
        response.append(_server_migrate(cs, server))
    utils.print_list(response, ['Server UUID', 'Migration Accepted', 'Error Message'])