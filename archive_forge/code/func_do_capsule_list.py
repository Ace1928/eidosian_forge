import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import template_utils
from zunclient.common import utils as zun_utils
from zunclient.i18n import _
@utils.arg('--all-projects', action='store_true', default=False, help='List containers in all projects')
@utils.arg('--marker', metavar='<marker>', default=None, help='The last container UUID of the previous page; displays list of containers after "marker".')
@utils.arg('--limit', metavar='<limit>', type=int, help='Maximum number of containers to return')
@utils.arg('--sort-key', metavar='<sort-key>', help='Column to sort results by')
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
def do_capsule_list(cs, args):
    """Print a list of available capsules."""
    opts = {}
    opts['all_projects'] = args.all_projects
    opts['marker'] = args.marker
    opts['limit'] = args.limit
    opts['sort_key'] = args.sort_key
    opts['sort_dir'] = args.sort_dir
    opts = zun_utils.remove_null_parms(**opts)
    capsules = cs.capsules.list(**opts)
    zun_utils.list_capsules(capsules)