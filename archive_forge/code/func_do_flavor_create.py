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
@utils.arg('name', metavar='<name>', help=_('Unique name of the new flavor.'))
@utils.arg('id', metavar='<id>', help=_("Unique ID of the new flavor. Specifying 'auto' will generated a UUID for the ID."))
@utils.arg('ram', metavar='<ram>', help=_('Memory size in MiB.'))
@utils.arg('disk', metavar='<disk>', help=_('Disk size in GiB.'))
@utils.arg('--ephemeral', metavar='<ephemeral>', help=_('Ephemeral space size in GiB (default 0).'), default=0)
@utils.arg('vcpus', metavar='<vcpus>', help=_('Number of vcpus'))
@utils.arg('--swap', metavar='<swap>', help=_('Additional swap space size in MiB (default 0).'), default=0)
@utils.arg('--rxtx-factor', metavar='<factor>', help=_('RX/TX factor (default 1).'), default=1.0)
@utils.arg('--is-public', metavar='<is-public>', help=_('Make flavor accessible to the public (default true).'), type=lambda v: strutils.bool_from_string(v, True), default=True)
@utils.arg('--description', metavar='<description>', help=_('A free form description of the flavor. Limited to 65535 characters in length. Only printable characters are allowed.'), start_version='2.55')
def do_flavor_create(cs, args):
    """Create a new flavor."""
    if cs.api_version >= api_versions.APIVersion('2.55'):
        description = args.description
    else:
        description = None
    f = cs.flavors.create(args.name, args.ram, args.vcpus, args.disk, args.id, args.ephemeral, args.swap, args.rxtx_factor, args.is_public, description)
    _print_flavor_list(cs, [f])