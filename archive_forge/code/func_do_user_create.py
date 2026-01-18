import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('name', metavar='<name>', help=_('Name of user.'))
@utils.arg('password', metavar='<password>', help=_('Password of user.'))
@utils.arg('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
@utils.arg('--databases', metavar='<databases>', help=_('Optional list of databases.'), nargs='+', default=[])
@utils.service_type('database')
def do_user_create(cs, args):
    """Creates a user on an instance."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    databases = [{'name': value} for value in args.databases]
    user = {'name': args.name, 'password': args.password, 'databases': databases}
    if args.host:
        user['host'] = args.host
    cs.users.create(instance, [user])