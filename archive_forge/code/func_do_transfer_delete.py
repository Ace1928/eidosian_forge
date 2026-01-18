import argparse
import collections
import copy
import os
from oslo_utils import strutils
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3 import availability_zones
@utils.arg('transfer', metavar='<transfer>', help='Name or ID of transfer to delete.')
def do_transfer_delete(cs, args):
    """Undoes a transfer."""
    transfer = shell_utils.find_transfer(cs, args.transfer)
    transfer.delete()