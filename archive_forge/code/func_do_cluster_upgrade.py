import argparse
import sys
import time
from troveclient.i18n import _
from troveclient import exceptions
from troveclient import utils
from troveclient.v1 import modules
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster.'))
@utils.arg('datastore_version', metavar='<datastore_version>', help=_('A datastore version name or ID.'))
@utils.service_type('database')
def do_cluster_upgrade(cs, args):
    """Upgrades a cluster to a new datastore version."""
    cluster = _find_cluster(cs, args.cluster)
    cs.clusters.upgrade(cluster, args.datastore_version)