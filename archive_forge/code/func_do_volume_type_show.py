import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('volume_type', metavar='<volume_type>', help='ID or name of the volume type.')
@utils.service_type('database')
def do_volume_type_show(cs, args):
    """Shows details of a volume type."""
    volume_type = _find_volume_type(cs, args.volume_type)
    _print_object(volume_type)