from magnumclient.common import cliutils as utils
from magnumclient.common import utils as magnum_utils
from magnumclient.exceptions import InvalidAttribute
from magnumclient.i18n import _
from magnumclient.v1 import basemodels
@utils.arg('--limit', metavar='<limit>', type=int, help=_('Maximum number of cluster templates to return'))
@utils.arg('--sort-key', metavar='<sort-key>', help=_('Column to sort results by'))
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help=_('Direction to sort. "asc" or "desc".'))
@utils.arg('--fields', default=None, metavar='<fields>', help=_('Comma-separated list of fields to display. Available fields: uuid, name, coe, image_id, public, link, apiserver_port, server_type, tls_disabled, registry_enabled'))
@utils.arg('--detail', action='store_true', default=False, help=_('Show detailed information about the cluster templates.'))
@utils.deprecated(utils.MAGNUM_CLIENT_DEPRECATION_WARNING)
def do_cluster_template_list(cs, args):
    """Print a list of cluster templates."""
    nodes = cs.cluster_templates.list(limit=args.limit, sort_key=args.sort_key, sort_dir=args.sort_dir, detail=args.detail)
    if args.detail:
        columns = basemodels.OUTPUT_ATTRIBUTES
    else:
        columns = ['uuid', 'name']
    columns += utils._get_list_table_columns_and_formatters(args.fields, nodes, exclude_fields=(c.lower() for c in columns))[0]
    utils.print_list(nodes, columns, {'versions': magnum_utils.print_list_field('versions')}, sortby_index=None)