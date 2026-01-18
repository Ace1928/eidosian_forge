import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.service_type('database')
def do_user_list(cs, args):
    """Lists the users for an instance."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    items = cs.users.list(instance)
    users = items
    while items.next:
        items = cs.users.list(instance, marker=items.next)
        users += items
    for user in users:
        db_names = [db['name'] for db in user.databases]
        user.databases = ', '.join(db_names)
    utils.print_list(users, ['name', 'host', 'databases'])