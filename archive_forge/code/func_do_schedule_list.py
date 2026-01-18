import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_schedule_list(cs, args):
    """Lists scheduled backups for an instance."""
    instance = _find_instance(cs, args.instance)
    schedules = cs.backups.schedule_list(instance)
    utils.print_list(schedules, ['id', 'name', 'pattern', 'next_execution_time'], order_by='next_execution_time')