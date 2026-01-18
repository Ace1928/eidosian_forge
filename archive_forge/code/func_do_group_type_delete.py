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
@api_versions.wraps('3.11')
@utils.arg('group_type', metavar='<group_type>', nargs='+', help='Name or ID of group type or types to delete.')
def do_group_type_delete(cs, args):
    """Deletes group type or types."""
    failure_count = 0
    for group_type in args.group_type:
        try:
            gtype = shell_utils.find_group_type(cs, group_type)
            cs.group_types.delete(gtype)
            print('Request to delete group type %s has been accepted.' % group_type)
        except Exception as e:
            failure_count += 1
            print('Delete for group type %s failed: %s' % (group_type, e))
    if failure_count == len(args.group_type):
        raise exceptions.CommandError('Unable to delete any of the specified types.')