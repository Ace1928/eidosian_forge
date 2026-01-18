import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('module', metavar='<module>', type=str, help=_('ID or name of the module.'))
@utils.arg('--include_clustered', action='store_true', default=False, help=_('Include instances that are part of a cluster (default %(default)s).'))
@utils.service_type('database')
def do_module_instance_count(cs, args):
    """Lists a count of the instances for each module md5."""
    module = _find_module(cs, args.module)
    count_list = cs.modules.instances(module, include_clustered=args.include_clustered, count_only=True)
    field_list = ['module_name', 'min_updated_date', 'max_updated_date', 'module_md5', 'current', 'instance_count']
    utils.print_list(count_list, field_list, labels={'module_md5': 'Module MD5', 'instance_count': 'Count', 'module_id': 'Module ID'})