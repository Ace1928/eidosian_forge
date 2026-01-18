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
@utils.arg('server', metavar='<server>', help=_('Name or ID of server.'))
@utils.arg('host', metavar='<host>', nargs='?', help=_('Name or ID of the target host.  If no host is specified, the scheduler will choose one.'))
@utils.arg('--password', dest='password', metavar='<password>', help=_('Set the provided admin password on the evacuated server. Not applicable if the server is on shared storage.'))
@utils.arg('--on-shared-storage', dest='on_shared_storage', action='store_true', default=False, help=_('Specifies whether server files are located on shared storage.'), start_version='2.0', end_version='2.13')
@utils.arg('--force', dest='force', action='store_true', default=False, help=_('Force an evacuation by not verifying the provided destination host by the scheduler. WARNING: This could result in failures to actually evacuate the server to the specified host. It is recommended to either not specify a host so that the scheduler will pick one, or specify a host without --force.'), start_version='2.29', end_version='2.67')
def do_evacuate(cs, args):
    """Evacuate server from failed host."""
    server = _find_server(cs, args.server)
    on_shared_storage = getattr(args, 'on_shared_storage', None)
    force = getattr(args, 'force', None)
    update_kwargs = {}
    if on_shared_storage is not None:
        update_kwargs['on_shared_storage'] = on_shared_storage
    if force:
        update_kwargs['force'] = force
    res = server.evacuate(host=args.host, password=args.password, **update_kwargs)[1]
    if isinstance(res, dict):
        utils.print_dict(res)