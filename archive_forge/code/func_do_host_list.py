import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('--marker', metavar='<marker>', default=None, help='The last host UUID of the previous page; displays list of hosts after "marker".')
@utils.arg('--limit', metavar='<limit>', type=int, help='Maximum number of hosts to return')
@utils.arg('--sort-key', metavar='<sort-key>', help='Column to sort results by')
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
def do_host_list(cs, args):
    """Print a list of available host."""
    opts = {}
    opts['marker'] = args.marker
    opts['limit'] = args.limit
    opts['sort_key'] = args.sort_key
    opts['sort_dir'] = args.sort_dir
    opts = zun_utils.remove_null_parms(**opts)
    hosts = cs.hosts.list(**opts)
    columns = ('uuid', 'hostname', 'mem_total', 'cpus', 'disk_total')
    utils.print_list(hosts, columns, {'versions': zun_utils.print_list_field('versions')}, sortby_index=None)