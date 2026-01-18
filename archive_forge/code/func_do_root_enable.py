import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('instance_or_cluster', metavar='<instance_or_cluster>', help=_('ID or name of the instance or cluster.'))
@utils.arg('--root_password', metavar='<root_password>', default=None, help=_('Root password to set.'))
@utils.service_type('database')
def do_root_enable(cs, args):
    """Enables root for an instance and resets if already exists."""
    instance_or_cluster, resource_type = _find_instance_or_cluster(cs, args.instance_or_cluster)
    if resource_type == 'instance':
        root = cs.root.create_instance_root(instance_or_cluster, args.root_password)
    else:
        root = cs.root.create_cluster_root(instance_or_cluster, args.root_password)
    utils.print_dict({'name': root[0], 'password': root[1]})