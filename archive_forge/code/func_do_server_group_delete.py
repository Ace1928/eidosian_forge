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
@utils.arg('id', metavar='<id>', nargs='+', help=_('Unique ID(s) of the server group to delete.'))
def do_server_group_delete(cs, args):
    """Delete specific server group(s)."""
    failure_count = 0
    for sg in args.id:
        try:
            cs.server_groups.delete(sg)
            print(_('Server group %s has been successfully deleted.') % sg)
        except Exception as e:
            failure_count += 1
            print(_('Delete for server group %(sg)s failed: %(e)s') % {'sg': sg, 'e': e})
    if failure_count == len(args.id):
        raise exceptions.CommandError(_('Unable to delete any of the specified server groups.'))