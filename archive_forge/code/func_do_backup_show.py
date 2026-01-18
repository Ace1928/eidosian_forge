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
@utils.arg('backup', metavar='<backup>', help='Name or ID of backup.')
def do_backup_show(cs, args):
    """Shows backup details."""
    backup = shell_utils.find_backup(cs, args.backup)
    info = dict()
    info.update(backup._info)
    info.pop('links', None)
    shell_utils.print_dict(info)