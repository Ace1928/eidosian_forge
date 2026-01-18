import os
from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient import exceptions
from magnumclient.i18n import _
@utils.arg('cluster', metavar='<cluster>', help=_('UUID or name of cluster'))
@utils.arg('--rollback', action='store_true', default=False, help=_('Rollback cluster on update failure.'))
@utils.arg('op', metavar='<op>', choices=['add', 'replace', 'remove'], help=_("Operations: 'add', 'replace' or 'remove'"))
@utils.arg('attributes', metavar='<path=value>', nargs='+', action='append', default=[], help=_('Attributes to add/replace or remove (only PATH is necessary on remove)'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_update(cs, args):
    """Update information about the given cluster."""
    if args.rollback and args.magnum_api_version and (args.magnum_api_version in ('1.0', '1.1', '1.2')):
        raise exceptions.CommandError('Rollback is not supported in API v%s. Please use API v1.3+.' % args.magnum_api_version)
    patch = magnum_utils.args_array_to_patch(args.op, args.attributes[0])
    try:
        cluster = cs.clusters.update(args.cluster, patch, args.rollback)
    except Exception as e:
        print('ERROR: %s' % e.details)
        return
    if args.magnum_api_version and args.magnum_api_version == '1.1':
        _show_cluster(cluster)
    else:
        print('Request to update cluster %s has been accepted.' % args.cluster)