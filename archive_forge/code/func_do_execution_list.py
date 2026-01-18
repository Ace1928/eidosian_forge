import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('id', metavar='<schedule id>', help=_('Id of the schedule.'))
@utils.arg('--limit', metavar='<limit>', default=None, type=int, help=_('Return up to N number of the most recent executions.'))
@utils.arg('--marker', metavar='<ID>', type=str, default=None, help=_('Begin displaying the results for IDs greater than the specified marker. When used with --limit, set this to the last ID displayed in the previous run.'))
@utils.service_type('database')
def do_execution_list(cs, args):
    """Lists executions of a scheduled backup of an instance."""
    executions = cs.backups.execution_list(args.id, marker=args.marker, limit=args.limit)
    utils.print_list(executions, ['id', 'created_at', 'state', 'output'], labels={'created_at': 'Execution Time'}, order_by='created_at')