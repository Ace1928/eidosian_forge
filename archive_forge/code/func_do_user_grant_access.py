import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('name', metavar='<name>', help=_('Name of user.'))
@utils.arg('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
@utils.arg('databases', metavar='<databases>', help=_('List of databases.'), nargs='+', default=[])
@utils.service_type('database')
def do_user_grant_access(cs, args):
    """Grants access to a database(s) for a user."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    cs.users.grant(instance, args.name, args.databases, hostname=args.host)