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
@utils.arg('host', metavar='<host>', default=None, nargs='?', help=_('Destination host name. If no host is specified, the scheduler will choose one.'))
@utils.arg('--block-migrate', action='store_true', dest='block_migrate', default=False, help=_('True in case of block_migration. (Default=False:live_migration)'), start_version='2.0', end_version='2.24')
@utils.arg('--block-migrate', action='store_true', dest='block_migrate', default='auto', help=_('True in case of block_migration. (Default=auto:live_migration)'), start_version='2.25')
@utils.arg('--disk-over-commit', action='store_true', dest='disk_over_commit', default=False, help=_('Allow overcommit. (Default=False)'), start_version='2.0', end_version='2.24')
@utils.arg('--force', dest='force', action='store_true', default=False, help=_('Force a live-migration by not verifying the provided destination host by the scheduler. WARNING: This could result in failures to actually live migrate the server to the specified host. It is recommended to either not specify a host so that the scheduler will pick one, or specify a host without --force.'), start_version='2.30', end_version='2.67')
def do_live_migration(cs, args):
    """Migrate running server to a new machine."""
    update_kwargs = {}
    if 'disk_over_commit' in args:
        update_kwargs['disk_over_commit'] = args.disk_over_commit
    if 'force' in args and args.force:
        update_kwargs['force'] = args.force
    _find_server(cs, args.server).live_migrate(args.host, args.block_migrate, **update_kwargs)