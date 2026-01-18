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
@utils.arg('--group-snapshot', metavar='<group-snapshot>', help='Name or ID of a group snapshot. Default=None.')
@utils.arg('--source-group', metavar='<source-group>', help='Name or ID of a source group. Default=None.')
@utils.arg('--name', metavar='<name>', help='Name of a group. Default=None.')
@utils.arg('--description', metavar='<description>', help='Description of a group. Default=None.')
def do_group_create_from_src(cs, args):
    """Creates a group from a group snapshot or a source group."""
    if not args.group_snapshot and (not args.source_group):
        msg = 'Cannot create group because neither group snapshot nor source group is provided.'
        raise exceptions.ClientException(code=1, message=msg)
    if args.group_snapshot and args.source_group:
        msg = 'Cannot create group because both group snapshot and source group are provided.'
        raise exceptions.ClientException(code=1, message=msg)
    group_snapshot = None
    if args.group_snapshot:
        group_snapshot = shell_utils.find_group_snapshot(cs, args.group_snapshot)
    source_group = None
    if args.source_group:
        source_group = shell_utils.find_group(cs, args.source_group)
    info = cs.groups.create_from_src(group_snapshot.id if group_snapshot else None, source_group.id if source_group else None, args.name, args.description)
    info.pop('links', None)
    shell_utils.print_dict(info)