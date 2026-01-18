import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
@utils.deprecation_map(DEPRECATING_PARAMS)
@utils.arg('positional_name', metavar='<name>', nargs='?', default=None, help=_('Name of the cluster to create.'))
@utils.arg('--name', metavar='<name>', default=None, help=_('Name of the cluster to create. %s') % utils.NAME_DEPRECATION_HELP)
@utils.arg('--cluster-template', required=True, metavar='<cluster_template>', help=_('ID or name of the cluster template.'))
@utils.arg('--keypair-id', dest='keypair', metavar='<keypair>', default=None, help=utils.deprecation_message('Name of the keypair to use for this cluster.', 'keypair'))
@utils.arg('--keypair', dest='keypair', metavar='<keypair>', default=None, help=_('Name of the keypair to use for this cluster.'))
@utils.arg('--docker-volume-size', metavar='<docker-volume-size>', type=int, help=_('The size in GB for the docker volume to use'))
@utils.arg('--labels', metavar='<KEY1=VALUE1,KEY2=VALUE2;KEY3=VALUE3...>', action='append', help=_('Arbitrary labels in the form of key=value pairs to associate with a cluster. May be used multiple times.'))
@utils.arg('--node-count', metavar='<node-count>', type=int, default=1, help=_('The cluster node count.'))
@utils.arg('--master-count', metavar='<master-count>', type=int, default=1, help=_('The number of master nodes for the cluster.'))
@utils.arg('--discovery-url', metavar='<discovery-url>', help=_('Specifies custom discovery url for node discovery.'))
@utils.arg('--timeout', metavar='<timeout>', type=int, default=60, help=_('The timeout for cluster creation in minutes. The default is 60 minutes.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_create(cs, args):
    """Create a cluster."""
    args.command = 'cluster-create'
    utils.validate_name_args(args.positional_name, args.name)
    cluster_template = cs.cluster_templates.get(args.cluster_template)
    opts = dict()
    opts['name'] = args.positional_name or args.name
    opts['cluster_template_id'] = cluster_template.uuid
    opts['keypair'] = args.keypair
    if args.docker_volume_size is not None:
        opts['docker_volume_size'] = args.docker_volume_size
    if args.labels is not None:
        opts['labels'] = magnum_utils.handle_labels(args.labels)
    opts['node_count'] = args.node_count
    opts['master_count'] = args.master_count
    opts['discovery_url'] = args.discovery_url
    opts['create_timeout'] = args.timeout
    try:
        cluster = cs.clusters.create(**opts)
        if args.magnum_api_version and args.magnum_api_version == '1.1':
            _show_cluster(cluster)
        else:
            uuid = str(cluster._info['uuid'])
            print('Request to create cluster %s has been accepted.' % uuid)
    except Exception as e:
        print('Create for cluster %s failed: %s' % (opts['name'], e))