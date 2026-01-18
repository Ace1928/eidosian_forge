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
@utils.arg('group', metavar='<group>', help='Name or ID of a group.')
@utils.arg('--name', metavar='<name>', help='New name for group. Default=None.')
@utils.arg('--description', metavar='<description>', help='New description for group. Default=None.')
@utils.arg('--add-volumes', metavar='<uuid1,uuid2,......>', help='UUID of one or more volumes to be added to the group, separated by commas. Default=None.')
@utils.arg('--remove-volumes', metavar='<uuid3,uuid4,......>', help='UUID of one or more volumes to be removed from the group, separated by commas. Default=None.')
def do_group_update(cs, args):
    """Updates a group."""
    kwargs = {}
    if args.name is not None:
        kwargs['name'] = args.name
    if args.description is not None:
        kwargs['description'] = args.description
    if args.add_volumes is not None:
        kwargs['add_volumes'] = args.add_volumes
    if args.remove_volumes is not None:
        kwargs['remove_volumes'] = args.remove_volumes
    if not kwargs:
        msg = 'At least one of the following args must be supplied: name, description, add-volumes, remove-volumes.'
        raise exceptions.ClientException(code=1, message=msg)
    shell_utils.find_group(cs, args.group).update(**kwargs)
    print("Request to update group '%s' has been accepted." % args.group)