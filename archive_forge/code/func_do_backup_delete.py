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
@utils.arg('--force', action='store_true', help='Allows deleting backup of a volume when its status is other than "available" or "error". Default=False.')
@utils.arg('backup', metavar='<backup>', nargs='+', help='Name or ID of backup(s) to delete.')
def do_backup_delete(cs, args):
    """Removes one or more backups."""
    failure_count = 0
    for backup in args.backup:
        try:
            shell_utils.find_backup(cs, backup).delete(args.force)
            print('Request to delete backup %s has been accepted.' % backup)
        except Exception as e:
            failure_count += 1
            print('Delete for backup %s failed: %s' % (backup, e))
    if failure_count == len(args.backup):
        raise exceptions.CommandError('Unable to delete any of the specified backups.')