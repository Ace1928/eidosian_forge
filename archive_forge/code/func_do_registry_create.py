import yaml
from oslo_serialization import jsonutils
from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
from zunclient import exceptions as exc
@utils.arg('--name', metavar='<name>', help='The name of the registry.')
@utils.arg('--username', metavar='<username>', help='The username to login to the registry.')
@utils.arg('--password', metavar='<password>', help='The password to login to the registry.')
@utils.arg('--domain', metavar='<domain>', required=True, help='The domain of the registry.')
def do_registry_create(cs, args):
    """Create a registry."""
    opts = {}
    opts['name'] = args.name
    opts['domain'] = args.domain
    opts['username'] = args.username
    opts['password'] = args.password
    opts = zun_utils.remove_null_parms(**opts)
    _show_registry(cs.registries.create(**opts))