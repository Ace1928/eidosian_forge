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
@api_versions.wraps('3.3')
@utils.arg('message', metavar='<message>', nargs='+', help='ID of one or more message to be deleted.')
def do_message_delete(cs, args):
    """Removes one or more messages."""
    failure_count = 0
    for message in args.message:
        try:
            shell_utils.find_message(cs, message).delete()
        except Exception as e:
            failure_count += 1
            print('Delete for message %s failed: %s' % (message, e))
    if failure_count == len(args.message):
        raise exceptions.CommandError('Unable to delete any of the specified messages.')