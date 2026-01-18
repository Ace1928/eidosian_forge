import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_id', metavar='<instance_id>', help=_('UUID for instance.'))
@utils.arg('key', metavar='<key>', help=_('Key to replace.'))
@utils.arg('value', metavar='<value>', help=_('New value to assign to <key>.'))
@utils.service_type('database')
def do_metadata_edit(cs, args):
    """Replaces metadata value with a new one, this is non-destructive."""
    cs.metadata.edit(args.instance_id, args.key, args.value)