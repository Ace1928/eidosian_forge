from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('share_group', metavar='<share_group>', help='Name or ID of the share group.')
@cliutils.arg('--name', metavar='<name>', help='Optional share group snapshot name. (Default=None)', default=None)
@cliutils.arg('--description', metavar='<description>', help='Optional share group snapshot description. (Default=None)', default=None)
@cliutils.service_type('sharev2')
def do_share_group_snapshot_create(cs, args):
    """Creates a new share group snapshot."""
    kwargs = {'name': args.name, 'description': args.description}
    share_group = _find_share_group(cs, args.share_group)
    sg_snapshot = cs.share_group_snapshots.create(share_group.id, **kwargs)
    _print_share_group_snapshot(cs, sg_snapshot)