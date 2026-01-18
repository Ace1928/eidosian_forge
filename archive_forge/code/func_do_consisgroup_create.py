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
@utils.arg('volumetypes', metavar='<volume-types>', help='Volume types.')
@utils.arg('--name', metavar='<name>', help='Name of a consistency group.')
@utils.arg('--description', metavar='<description>', default=None, help='Description of a consistency group. Default=None.')
@utils.arg('--availability-zone', metavar='<availability-zone>', default=None, help='Availability zone for volume. Default=None.')
def do_consisgroup_create(cs, args):
    """Creates a consistency group."""
    consistencygroup = cs.consistencygroups.create(args.volumetypes, args.name, args.description, availability_zone=args.availability_zone)
    info = dict()
    consistencygroup = cs.consistencygroups.get(consistencygroup.id)
    info.update(consistencygroup._info)
    info.pop('links', None)
    shell_utils.print_dict(info)