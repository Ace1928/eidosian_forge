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
@utils.arg('cgsnapshot', metavar='<cgsnapshot>', help='Name or ID of cgsnapshot.')
def do_cgsnapshot_show(cs, args):
    """Shows cgsnapshot details."""
    info = dict()
    cgsnapshot = shell_utils.find_cgsnapshot(cs, args.cgsnapshot)
    info.update(cgsnapshot._info)
    info.pop('links', None)
    shell_utils.print_dict(info)