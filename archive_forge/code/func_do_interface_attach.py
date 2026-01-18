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
@utils.arg('--port-id', metavar='<port_id>', help=_('Port ID.'), dest='port_id')
@utils.arg('--net-id', metavar='<net_id>', help=_('Network ID'), default=None, dest='net_id')
@utils.arg('--fixed-ip', metavar='<fixed_ip>', help=_('Requested fixed IP.'), default=None, dest='fixed_ip')
@utils.arg('--tag', metavar='<tag>', default=None, dest='tag', help=_('Tag for the attached interface.'), start_version='2.49')
def do_interface_attach(cs, args):
    """Attach a network interface to a server."""
    server = _find_server(cs, args.server)
    update_kwargs = {}
    if 'tag' in args and args.tag:
        update_kwargs['tag'] = args.tag
    network_interface = server.interface_attach(args.port_id, args.net_id, args.fixed_ip, **update_kwargs)
    _print_interface(network_interface)