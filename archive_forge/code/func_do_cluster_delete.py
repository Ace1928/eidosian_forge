import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
@utils.arg('cluster', metavar='<cluster>', nargs='+', help=_('ID or name of the (cluster)s to delete.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_delete(cs, args):
    """Delete specified cluster."""
    for id in args.cluster:
        try:
            cs.clusters.delete(id)
            print('Request to delete cluster %s has been accepted.' % id)
        except Exception as e:
            print('Delete for cluster %(cluster)s failed: %(e)s' % {'cluster': id, 'e': e})