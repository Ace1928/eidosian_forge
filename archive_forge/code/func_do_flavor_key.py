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
@utils.arg('flavor', metavar='<flavor>', help=_('Name or ID of flavor.'))
@utils.arg('action', metavar='<action>', choices=['set', 'unset'], help=_("Actions: 'set' or 'unset'."))
@utils.arg('metadata', metavar='<key=value>', nargs='+', action='append', default=[], help=_('Extra_specs to set/unset (only key is necessary on unset).'))
def do_flavor_key(cs, args):
    """Set or unset extra_spec for a flavor."""
    flavor = _find_flavor(cs, args.flavor)
    keypair = _extract_metadata(args)
    if args.action == 'set':
        flavor.set_keys(keypair)
    elif args.action == 'unset':
        flavor.unset_keys(keypair.keys())