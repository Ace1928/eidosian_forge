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
@utils.arg('backup', metavar='<backup>', nargs='+', help='Name or ID of the backup to modify.')
@utils.arg('--state', metavar='<state>', default='available', help='The state to assign to the backup. Valid values are "available", "error". Default=available.')
def do_backup_reset_state(cs, args):
    """Explicitly updates the backup state."""
    failure_count = 0
    single = len(args.backup) == 1
    for backup in args.backup:
        try:
            shell_utils.find_backup(cs, backup).reset_state(args.state)
            print("Request to update backup '%s' has been accepted." % backup)
        except Exception as e:
            failure_count += 1
            msg = 'Reset state for backup %s failed: %s' % (backup, e)
            if not single:
                print(msg)
    if failure_count == len(args.backup):
        if not single:
            msg = 'Unable to reset the state for any of the specified backups.'
        raise exceptions.CommandError(msg)