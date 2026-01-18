import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
@utils.arg('registry', metavar='<registry>', help='ID or name of the registry to show.')
@utils.arg('-f', '--format', metavar='<format>', action='store', choices=['json', 'yaml', 'table'], default='table', help='Print representation of the container.The choices of the output format is json,table,yaml.Defaults to table.')
def do_registry_show(cs, args):
    """Show details of a registry."""
    opts = {}
    opts['id'] = args.registry
    opts = zun_utils.remove_null_parms(**opts)
    registry = cs.registries.get(**opts)
    if args.format == 'json':
        print(jsonutils.dumps(registry._info, indent=4, sort_keys=True))
    elif args.format == 'yaml':
        print(yaml.safe_dump(registry._info, default_flow_style=False))
    elif args.format == 'table':
        _show_registry(registry)