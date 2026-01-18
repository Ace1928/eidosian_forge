import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_root_disable(cs, args):
    """Disables root for an instance."""
    instance = _find_instance(cs, args.instance)
    cs.root.disable_instance_root(instance)