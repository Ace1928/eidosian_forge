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
@utils.arg('transfer', metavar='<transfer>', help='Name or ID of transfer to accept.')
def do_transfer_show(cs, args):
    """Shows transfer details."""
    transfer = shell_utils.find_transfer(cs, args.transfer)
    info = dict()
    info.update(transfer._info)
    info.pop('links', None)
    shell_utils.print_dict(info)