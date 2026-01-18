import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('id', metavar='<schedule id>', help=_('Id of the schedule.'))
@utils.service_type('database')
def do_schedule_delete(cs, args):
    """Deletes a schedule."""
    cs.backups.schedule_delete(args.id)