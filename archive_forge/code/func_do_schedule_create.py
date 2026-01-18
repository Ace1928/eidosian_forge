import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('pattern', metavar='<pattern>', help=_('Cron style pattern describing schedule occurrence.'))
@utils.arg('name', metavar='<name>', help=_('Name of the backup.'))
@utils.arg('--description', metavar='<description>', default=None, help=_('An optional description for the backup.'))
@utils.arg('--incremental', action='store_true', default=False, help=_('Flag to select incremental backup based on most recent backup.'))
@utils.service_type('database')
def do_schedule_create(cs, args):
    """Schedules backups for an instance."""
    instance = _find_instance(cs, args.instance)
    backup = cs.backups.schedule_create(instance, args.pattern, args.name, description=args.description, incremental=args.incremental)
    _print_object(backup)