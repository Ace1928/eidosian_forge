import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_configuration_default(cs, args):
    """Shows the default configuration of an instance."""
    instance = _find_instance(cs, args.instance)
    configs = cs.instances.configuration(instance)
    utils.print_dict(configs._info['configuration'])