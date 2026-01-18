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
@utils.arg('--console-type', default='serial', help=_('Type of serial console, default="serial".'))
def do_get_serial_console(cs, args):
    """Get a serial console to a server."""
    if args.console_type not in ('serial',):
        raise exceptions.CommandError(_("Invalid parameter value for 'console_type', currently supported 'serial'."))
    server = _find_server(cs, args.server)
    data = server.get_serial_console(args.console_type)
    print_console(cs, data)