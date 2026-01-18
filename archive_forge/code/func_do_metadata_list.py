import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_id', metavar='<instance_id>', help=_('UUID for instance.'))
@utils.service_type('database')
def do_metadata_list(cs, args):
    """Shows all metadata for instance <id>."""
    result = cs.metadata.list(args.instance_id)
    _print_object(result)