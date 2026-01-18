import argparse
import collections
import os
from oslo_utils import strutils
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import exceptions
from cinderclient import shell_utils
from cinderclient import utils
from cinderclient.v3.shell_base import *  # noqa
from cinderclient.v3.shell_base import CheckSizeArgForCreate
@api_versions.wraps('3.7')
@utils.arg('binary', metavar='<binary>', nargs='?', default='cinder-volume', help='Binary to filter by.  Default: cinder-volume.')
@utils.arg('name', metavar='<cluster-name>', help='Name of the clustered services to update.')
def do_cluster_enable(cs, args):
    """Enables clustered services."""
    cluster = cs.clusters.update(args.name, args.binary, disabled=False)
    shell_utils.print_dict(cluster.to_dict())