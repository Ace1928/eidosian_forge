import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('--limit', metavar='<limit>', default=None, help=_('Return up to N number of the most recent backups.'))
@utils.arg('--marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than the specified marker. When used with --limit, set this to the last ID displayed in the previous run.'))
@utils.service_type('database')
def do_backup_list_instance(cs, args):
    """Lists available backups for an instance."""
    instance = _find_instance(cs, args.instance)
    items = cs.instances.backups(instance, limit=args.limit, marker=args.marker)
    backups = items
    while items.next and (not args.limit):
        items = cs.instances.backups(instance, marker=items.next)
        backups += items
    utils.print_list(backups, ['id', 'name', 'status', 'parent_id', 'updated'], order_by='updated')