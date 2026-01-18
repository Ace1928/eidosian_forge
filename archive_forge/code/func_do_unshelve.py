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
@utils.arg('--availability-zone', metavar='<availability-zone>', default=None, dest='availability_zone', help=_('Name of the availability zone in which to unshelve a SHELVED_OFFLOADED server.'), start_version='2.77')
def do_unshelve(cs, args):
    """Unshelve a server."""
    update_kwargs = {}
    if cs.api_version >= api_versions.APIVersion('2.77'):
        if 'availability_zone' in args and args.availability_zone is not None:
            update_kwargs['availability_zone'] = args.availability_zone
    server = _find_server(cs, args.server)
    server.unshelve(**update_kwargs)