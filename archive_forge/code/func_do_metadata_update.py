import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_id', metavar='<instance_id>', help=_('UUID for instance.'))
@utils.arg('key', metavar='<key>', help=_('Key to update.'))
@utils.arg('newkey', metavar='<newkey>', help=_('New key.'))
@utils.arg('value', metavar='<value>', help=_('Value to assign to <newkey>.'))
@utils.service_type('database')
def do_metadata_update(cs, args):
    """Updates metadata, this is destructive."""
    cs.metadata.update(args.instance_id, args.key, args.newkey, args.value)