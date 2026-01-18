import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance', metavar='<instance>', help=_('ID or name of the instance.'))
@utils.arg('name', metavar='<name>', help=_('Name of the database.'))
@utils.arg('--character_set', metavar='<character_set>', default=None, help=_('Optional character set for database.'))
@utils.arg('--collate', metavar='<collate>', default=None, help=_('Optional collation type for database.'))
@utils.service_type('database')
def do_database_create(cs, args):
    """Creates a database on an instance."""
    instance, _ = _find_instance_or_cluster(cs, args.instance)
    database_dict = {'name': args.name}
    if args.collate:
        database_dict['collate'] = args.collate
    if args.character_set:
        database_dict['character_set'] = args.character_set
    cs.databases.create(instance, [database_dict])