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
@utils.arg('backup', metavar='<backup>', help='ID of the backup to export.')
def do_backup_export(cs, args):
    """Export backup metadata record."""
    info = cs.backups.export_record(args.backup)
    shell_utils.print_dict(info)