import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.arg('size', metavar='<size>', type=int, default=None, help=_('New size of the instance disk volume in GB.'))
@utils.service_type('database')
def do_resize_volume(cs, args):
    """Resizes the volume size of an instance."""
    instance = _find_instance(cs, args.instance)
    cs.instances.resize_volume(instance, args.size)