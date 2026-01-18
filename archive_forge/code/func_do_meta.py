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
@utils.arg('action', metavar='<action>', choices=['set', 'delete'], help=_("Actions: 'set' or 'delete'."))
@utils.arg('metadata', metavar='<key=value>', nargs='+', action='append', default=[], help=_('Metadata to set or delete (only key is necessary on delete).'))
def do_meta(cs, args):
    """Set or delete metadata on a server."""
    server = _find_server(cs, args.server)
    metadata = _extract_metadata(args)
    if args.action == 'set':
        cs.servers.set_meta(server, metadata)
    elif args.action == 'delete':
        cs.servers.delete_meta(server, sorted(metadata.keys(), reverse=True))