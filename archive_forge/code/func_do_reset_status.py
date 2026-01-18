import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_reset_status(cs, args):
    """Set the task status of an instance to NONE if the instance is in BUILD
    or ERROR state. Resetting task status of an instance in BUILD state will
    allow the instance to be deleted.
    """
    instance = _find_instance(cs, args.instance)
    cs.instances.reset_status(instance=instance)