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
@utils.arg('backup_service', metavar='<backup_service>', help='Backup service to use for importing the backup.')
@utils.arg('backup_url', metavar='<backup_url>', help='Backup URL for importing the backup metadata.')
def do_backup_import(cs, args):
    """Import backup metadata record."""
    info = cs.backups.import_record(args.backup_service, args.backup_url)
    info.pop('links', None)
    shell_utils.print_dict(info)