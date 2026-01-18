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
@api_versions.wraps('3.9')
@utils.arg('backup', metavar='<backup>', help='Name or ID of backup to rename.')
@utils.arg('--name', nargs='?', metavar='<name>', help='New name for backup.')
@utils.arg('--description', metavar='<description>', help='Backup description. Default=None.')
@utils.arg('--metadata', nargs='*', metavar='<key=value>', default=None, help='Metadata key and value pairs. Default=None.', start_version='3.43')
def do_backup_update(cs, args):
    """Updates a backup."""
    kwargs = {}
    if args.name is not None:
        kwargs['name'] = args.name
    if args.description is not None:
        kwargs['description'] = args.description
    if cs.api_version >= api_versions.APIVersion('3.43'):
        if args.metadata is not None:
            kwargs['metadata'] = shell_utils.extract_metadata(args)
    if not kwargs:
        msg = 'Must supply at least one: name, description or metadata.'
        raise exceptions.ClientException(code=1, message=msg)
    shell_utils.find_backup(cs, args.backup).update(**kwargs)
    print("Request to update backup '%s' has been accepted." % args.backup)