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
@utils.arg('--reason', metavar='<reason>', help=_('Reason for locking the server.'), start_version='2.73')
def do_lock(cs, args):
    """Lock a server. A normal (non-admin) user will not be able to execute
    actions on a locked server.
    """
    update_kwargs = {}
    if 'reason' in args and args.reason is not None:
        update_kwargs['reason'] = args.reason
    server = _find_server(cs, args.server)
    server.lock(**update_kwargs)