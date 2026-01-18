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
@utils.arg('grouptype', metavar='<group-type>', help='Group type.')
@utils.arg('volumetypes', metavar='<volume-types>', help='Comma-separated list of volume types.')
@utils.arg('--name', metavar='<name>', help='Name of a group.')
@utils.arg('--description', metavar='<description>', default=None, help='Description of a group. Default=None.')
@utils.arg('--availability-zone', metavar='<availability-zone>', default=None, help='Availability zone for group. Default=None.')
def do_group_create(cs, args):
    """Creates a group."""
    group = cs.groups.create(args.grouptype, args.volumetypes, args.name, args.description, availability_zone=args.availability_zone)
    info = dict()
    group = cs.groups.get(group.id)
    info.update(group._info)
    info.pop('links', None)
    shell_utils.print_dict(info)
    with cs.groups.completion_cache('uuid', cinderclient.v3.groups.Group, mode='a'):
        cs.groups.write_to_completion_cache('uuid', group.id)
    if group.name is not None:
        with cs.groups.completion_cache('name', cinderclient.v3.groups.Group, mode='a'):
            cs.groups.write_to_completion_cache('name', group.name)