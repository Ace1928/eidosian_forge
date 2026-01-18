import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_database_list(cs, args):
    """Lists available databases on an instance."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    items = cs.databases.list(instance)
    databases = items
    while items.next:
        items = cs.databases.list(instance, marker=items.next)
        databases += items
    utils.print_list(databases, ['name'])