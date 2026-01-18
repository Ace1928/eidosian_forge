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
@utils.arg('--name', metavar='<name>', default=None, help='Cgsnapshot name. Default=None.')
@utils.arg('--description', metavar='<description>', default=None, help='Cgsnapshot description. Default=None.')
def do_cgsnapshot_create(cs, args):
    """Creates a cgsnapshot."""
    consistencygroup = shell_utils.find_consistencygroup(cs, args.consistencygroup)
    cgsnapshot = cs.cgsnapshots.create(consistencygroup.id, args.name, args.description)
    info = dict()
    cgsnapshot = cs.cgsnapshots.get(cgsnapshot.id)
    info.update(cgsnapshot._info)
    info.pop('links', None)
    shell_utils.print_dict(info)