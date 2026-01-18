import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.arg('modules', metavar='<module>', type=str, nargs='+', default=[], help=_('ID or name of the module.'))
@utils.service_type('database')
def do_module_apply(cs, args):
    """Apply modules to an instance."""
    instance = _find_instance(cs, args.instance)
    modules = []
    for module in args.modules:
        modules.append(_find_module(cs, module))
    result_list = cs.instances.module_apply(instance, modules)
    utils.print_list(result_list, ['name', 'type', 'datastore', 'datastore_version', 'status', 'message'], labels={'datastore_version': 'Version'})