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
@utils.arg('volume', metavar='<volume>', help='Cinder volume that already exists in the volume backend.')
@utils.arg('identifier', metavar='<identifier>', help='Name or other identifier for existing snapshot. This is backend specific.')
@utils.arg('--id-type', metavar='<id-type>', default='source-name', help='Type of backend device identifier provided, typically source-name or source-id (Default=source-name).')
@utils.arg('--name', metavar='<name>', help='Snapshot name (Default=None).')
@utils.arg('--description', metavar='<description>', help='Snapshot description (Default=None).')
@utils.arg('--metadata', nargs='*', metavar='<key=value>', help='Metadata key=value pairs (Default=None).')
def do_snapshot_manage(cs, args):
    """Manage an existing snapshot."""
    snapshot_metadata = None
    if args.metadata is not None:
        snapshot_metadata = shell_utils.extract_metadata(args)
    ref_dict = {args.id_type: args.identifier}
    if hasattr(args, 'source_name') and args.source_name is not None:
        ref_dict['source-name'] = args.source_name
    if hasattr(args, 'source_id') and args.source_id is not None:
        ref_dict['source-id'] = args.source_id
    volume = utils.find_volume(cs, args.volume)
    snapshot = cs.volume_snapshots.manage(volume_id=volume.id, ref=ref_dict, name=args.name, description=args.description, metadata=snapshot_metadata)
    info = {}
    snapshot = cs.volume_snapshots.get(snapshot.id)
    info.update(snapshot._info)
    info.pop('links', None)
    shell_utils.print_dict(info)