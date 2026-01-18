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
@api_versions.wraps('3.13')
@utils.arg('--list-volume', dest='list_volume', metavar='<False|True>', nargs='?', type=bool, const=True, default=False, help='Shows volumes included in the group.', start_version='3.25')
@utils.arg('group', metavar='<group>', help='Name or ID of a group.')
def do_group_show(cs, args):
    """Shows details of a group."""
    info = dict()
    if getattr(args, 'list_volume', None):
        group = shell_utils.find_group(cs, args.group, list_volume=args.list_volume)
    else:
        group = shell_utils.find_group(cs, args.group)
    info.update(group._info)
    info.pop('links', None)
    shell_utils.print_dict(info)