import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('host', metavar='<host>', help='Cinder host to show backend volume stats and properties; takes the form: host@backend-name')
def do_get_capabilities(cs, args):
    """Show backend volume stats and properties. Admin only."""
    capabilities = cs.capabilities.get(args.host)
    infos = dict()
    infos.update(capabilities._info)
    prop = infos.pop('properties', None)
    shell_utils.print_dict(infos, 'Volume stats')
    shell_utils.print_dict(prop, 'Backend properties', formatters=sorted(prop.keys()))