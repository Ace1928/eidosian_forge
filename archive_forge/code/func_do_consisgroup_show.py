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
@utils.arg('consistencygroup', metavar='<consistencygroup>', help='Name or ID of a consistency group.')
def do_consisgroup_show(cs, args):
    """Shows details of a consistency group."""
    info = dict()
    consistencygroup = shell_utils.find_consistencygroup(cs, args.consistencygroup)
    info.update(consistencygroup._info)
    info.pop('links', None)
    shell_utils.print_dict(info)