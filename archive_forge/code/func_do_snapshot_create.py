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
@api_versions.wraps('3.66')
@utils.arg('volume', metavar='<volume>', help='Name or ID of volume to snapshot.')
@utils.arg('--force', nargs='?', help=argparse.SUPPRESS)
@utils.arg('--name', metavar='<name>', default=None, help='Snapshot name. Default=None.')
@utils.arg('--display-name', help=argparse.SUPPRESS)
@utils.arg('--display_name', help=argparse.SUPPRESS)
@utils.arg('--description', metavar='<description>', default=None, help='Snapshot description. Default=None.')
@utils.arg('--display-description', help=argparse.SUPPRESS)
@utils.arg('--display_description', help=argparse.SUPPRESS)
@utils.arg('--metadata', nargs='*', metavar='<key=value>', default=None, help='Snapshot metadata key and value pairs. Default=None.')
def do_snapshot_create(cs, args):
    """Creates a snapshot."""
    if args.display_name is not None:
        args.name = args.display_name
    if args.display_description is not None:
        args.description = args.display_description
    snapshot_metadata = None
    if args.metadata is not None:
        snapshot_metadata = shell_utils.extract_metadata(args)
    force = getattr(args, 'force', None)
    volume = utils.find_volume(cs, args.volume)
    try:
        snapshot = cs.volume_snapshots.create(volume.id, force=force, name=args.name, description=args.description, metadata=snapshot_metadata)
    except ValueError as ve:
        em = cinderclient.v3.volume_snapshots.MV_3_66_FORCE_FLAG_ERROR
        if em == str(ve):
            raise exceptions.UnsupportedAttribute('force', start_version=None, end_version=api_versions.APIVersion('3.65'))
        else:
            raise
    shell_utils.print_volume_snapshot(snapshot)