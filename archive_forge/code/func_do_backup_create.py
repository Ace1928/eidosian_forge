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
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume to backup.')
@utils.arg('--container', metavar='<container>', default=None, help='Backup container name. Default=None.')
@utils.arg('--display-name', help=argparse.SUPPRESS)
@utils.arg('--name', metavar='<name>', default=None, help='Backup name. Default=None.')
@utils.arg('--display-description', help=argparse.SUPPRESS)
@utils.arg('--description', metavar='<description>', default=None, help='Backup description. Default=None.')
@utils.arg('--incremental', action='store_true', help='Incremental backup. Default=False.', default=False)
@utils.arg('--force', action='store_true', help='Allows or disallows backup of a volume when the volume is attached to an instance. If set to True, backs up the volume whether its status is "available" or "in-use". The backup of an "in-use" volume means your data is crash consistent. Default=False.', default=False)
@utils.arg('--snapshot-id', metavar='<snapshot-id>', default=None, help='ID of snapshot to backup. Default=None.')
def do_backup_create(cs, args):
    """Creates a volume backup."""
    if args.display_name is not None:
        args.name = args.display_name
    if args.display_description is not None:
        args.description = args.display_description
    volume = utils.find_volume(cs, args.volume)
    backup = cs.backups.create(volume.id, args.container, args.name, args.description, args.incremental, args.force, args.snapshot_id)
    info = {'volume_id': volume.id}
    info.update(backup._info)
    if 'links' in info:
        info.pop('links')
    shell_utils.print_dict(info)