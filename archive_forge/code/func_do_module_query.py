import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', type=str, help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_module_query(cs, args):
    """Query the status of the modules on an instance."""
    instance = _find_instance(cs, args.instance)
    result_list = cs.instances.module_query(instance)
    utils.print_list(result_list, ['name', 'type', 'datastore', 'datastore_version', 'status', 'message', 'created', 'updated'], labels={'datastore_version': 'Version'})