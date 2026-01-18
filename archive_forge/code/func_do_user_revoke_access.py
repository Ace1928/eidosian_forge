import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('name', metavar='<name>', help=_('Name of user.'))
@utils.arg('database', metavar='<database>', help=_('A single database.'))
@utils.arg('--host', metavar='<host>', default=None, help=_('Optional host of user.'))
@utils.service_type('database')
def do_user_revoke_access(cs, args):
    """Revokes access to a database for a user."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    cs.users.revoke(instance, args.name, args.database, hostname=args.host)