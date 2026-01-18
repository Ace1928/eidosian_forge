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
@utils.arg('--password', metavar='<password>', dest='password', help=_('The admin password to be set in the rescue environment.'))
@utils.arg('--image', metavar='<image>', dest='image', help=_('The image to rescue with.'))
def do_rescue(cs, args):
    """Reboots a server into rescue mode, which starts the machine
    from either the initial image or a specified image, attaching the current
    boot disk as secondary.
    """
    kwargs = {}
    if args.image:
        kwargs['image'] = _find_image(cs, args.image)
    if args.password:
        kwargs['password'] = args.password
    utils.print_dict(_find_server(cs, args.server).rescue(**kwargs)[1])