from magnumclient.common import cliutils as utils
from magnumclient.i18n import _
@utils.arg('--project-id', required=False, metavar='<project-id>', help=_('Project ID'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_stats_list(cs, args):
    """Show stats for the given project_id"""
    opts = {'project_id': args.project_id}
    stats = cs.stats.list(**opts)
    utils.print_dict(stats._info)