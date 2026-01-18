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
@utils.service_type('database')
def do_user_delete(cs, args):
    """Deletes a user from an instance."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    cs.users.delete(instance, args.name, hostname=args.host)