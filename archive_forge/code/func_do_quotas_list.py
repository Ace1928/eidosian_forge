from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.i18n import _
@utils.arg('--marker', metavar='<marker>', default=None, help=_('The last quota UUID of the previous page; displays list of quotas after "marker".'))
@utils.arg('--limit', metavar='<limit>', type=int, help=_('Maximum number of quotas to return.'))
@utils.arg('--sort-key', metavar='<sort-key>', help=_('Column to sort results by.'))
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help=_('Direction to sort. "asc" or "desc".'))
@utils.arg('--all-tenants', action='store_true', default=False, help=_('Flag to indicate list all tenant quotas.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_quotas_list(cs, args):
    """Print a list of available quotas."""
    quotas = cs.quotas.list(marker=args.marker, limit=args.limit, sort_key=args.sort_key, sort_dir=args.sort_dir, all_tenants=args.all_tenants)
    columns = ['project_id', 'resource', 'hard_limit']
    utils.print_list(quotas, columns, {'versions': magnum_utils.print_list_field('versions')}, sortby_index=None)