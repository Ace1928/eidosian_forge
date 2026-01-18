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
@utils.arg('--cgsnapshot', metavar='<cgsnapshot>', help='Name or ID of a cgsnapshot. Default=None.')
@utils.arg('--source-cg', metavar='<source-cg>', help='Name or ID of a source CG. Default=None.')
@utils.arg('--name', metavar='<name>', help='Name of a consistency group. Default=None.')
@utils.arg('--description', metavar='<description>', help='Description of a consistency group. Default=None.')
def do_consisgroup_create_from_src(cs, args):
    """Creates a consistency group from a cgsnapshot or a source CG."""
    if not args.cgsnapshot and (not args.source_cg):
        msg = 'Cannot create consistency group because neither cgsnapshot nor source CG is provided.'
        raise exceptions.ClientException(code=1, message=msg)
    if args.cgsnapshot and args.source_cg:
        msg = 'Cannot create consistency group because both cgsnapshot and source CG are provided.'
        raise exceptions.ClientException(code=1, message=msg)
    cgsnapshot = None
    if args.cgsnapshot:
        cgsnapshot = shell_utils.find_cgsnapshot(cs, args.cgsnapshot)
    source_cg = None
    if args.source_cg:
        source_cg = shell_utils.find_consistencygroup(cs, args.source_cg)
    info = cs.consistencygroups.create_from_src(cgsnapshot.id if cgsnapshot else None, source_cg.id if source_cg else None, args.name, args.description)
    info.pop('links', None)
    shell_utils.print_dict(info)