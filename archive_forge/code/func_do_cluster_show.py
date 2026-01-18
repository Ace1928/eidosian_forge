import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
@utils.arg('cluster', metavar='<cluster>', help=_('ID or name of the cluster to show.'))
@utils.arg('--long', action='store_true', default=False, help=_('Display extra associated cluster template info.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_show(cs, args):
    """Show details about the given cluster."""
    cluster = cs.clusters.get(args.cluster)
    if args.long:
        cluster_template = cs.cluster_templates.get(cluster.cluster_template_id)
        del cluster_template._info['links'], cluster_template._info['uuid']
        for key in cluster_template._info:
            if 'clustertemplate_' + key not in cluster._info:
                cluster._info['clustertemplate_' + key] = cluster_template._info[key]
    _show_cluster(cluster)