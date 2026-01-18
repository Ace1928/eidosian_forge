import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('datastore', metavar='<datastore>', help=_('ID or name of the datastore.'))
@utils.service_type('database')
def do_datastore_version_list(cs, args):
    """Lists available versions for a datastore."""
    datastore_versions = cs.datastore_versions.list(args.datastore)
    utils.print_list(datastore_versions, ['id', 'name'])