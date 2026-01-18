import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('--datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore. Optional if the ID of the datastore_version is provided.'))
@utils.arg('datastore_version', metavar='<datastore_version>', help=_('ID or name of the datastore version.'))
@utils.service_type('database')
def do_datastore_version_show(cs, args):
    """Shows details of a datastore version."""
    if args.datastore:
        datastore_version = cs.datastore_versions.get(args.datastore, args.datastore_version)
    elif utils.is_uuid_like(args.datastore_version):
        datastore_version = cs.datastore_versions.get_by_uuid(args.datastore_version)
    else:
        raise exceptions.NoUniqueMatch(_('The datastore name or id is required to retrieve a datastore version by name.'))
    _print_object(datastore_version)