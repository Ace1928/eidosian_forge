import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.14')
@utils.arg('group', metavar='<group>', help='Name or ID of a group.')
@utils.arg('--name', metavar='<name>', default=None, help='Group snapshot name. Default=None.')
@utils.arg('--description', metavar='<description>', default=None, help='Group snapshot description. Default=None.')
def do_group_snapshot_create(cs, args):
    """Creates a group snapshot."""
    group = shell_utils.find_group(cs, args.group)
    group_snapshot = cs.group_snapshots.create(group.id, args.name, args.description)
    info = dict()
    group_snapshot = cs.group_snapshots.get(group_snapshot.id)
    info.update(group_snapshot._info)
    info.pop('links', None)
    shell_utils.print_dict(info)