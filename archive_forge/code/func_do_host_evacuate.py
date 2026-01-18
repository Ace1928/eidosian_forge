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
@utils.arg('host', metavar='<host>', help='The hypervisor hostname (or pattern) to search for. WARNING: Use a fully qualified domain name if you only want to evacuate from a specific host.')
@utils.arg('--target_host', metavar='<target_host>', default=None, help=_('Name of target host. If no host is specified the scheduler will select a target.'))
@utils.arg('--on-shared-storage', dest='on_shared_storage', action='store_true', default=False, help=_('Specifies whether all instances files are on shared storage'), start_version='2.0', end_version='2.13')
@utils.arg('--force', dest='force', action='store_true', default=False, help=_('Force an evacuation by not verifying the provided destination host by the scheduler. WARNING: This could result in failures to actually evacuate the server to the specified host. It is recommended to either not specify a host so that the scheduler will pick one, or specify a host without --force.'), start_version='2.29', end_version='2.67')
@utils.arg('--strict', dest='strict', action='store_true', default=False, help=_('Evacuate host with exact hypervisor hostname match'))
def do_host_evacuate(cs, args):
    """Evacuate all instances from failed host."""
    response = []
    for server in _hyper_servers(cs, args.host, args.strict):
        response.append(_server_evacuate(cs, server, args))
    utils.print_list(response, ['Server UUID', 'Evacuate Accepted', 'Error Message'])