import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('execution', metavar='<execution>', help=_('Id of the execution to delete.'))
@utils.service_type('database')
def do_execution_delete(cs, args):
    """Deletes an execution."""
    cs.backups.execution_delete(args.execution)