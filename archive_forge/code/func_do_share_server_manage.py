from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@api_versions.wraps('2.49')
@cliutils.arg('host', metavar='<host>', type=str, help='Backend name as "<node_hostname>@<backend_name>".')
@cliutils.arg('share_network', metavar='<share_network>', help='Share network where share server has network allocations in.')
@cliutils.arg('identifier', metavar='<identifier>', type=str, help='A driver-specific share server identifier required by the driver to manage the share server.')
@cliutils.arg('--driver_options', '--driver-options', type=str, nargs='*', metavar='<key=value>', action='single_alias', help='One or more driver-specific key=value pairs that may be necessary to manage the share server (Optional, Default=None).', default=None)
@cliutils.arg('--share-network-subnet', '--share_network_subnet', type=str, metavar='<share_network_subnet>', help="Share network subnet where share server has network allocations in. The default subnet will be used if it's not specified. Available for microversion >= 2.51 (Optional, Default=None).", default=None)
@cliutils.arg('--wait', action='store_true', default='False', help='Wait for share server to manage')
def do_share_server_manage(cs, args):
    """Manage share server not handled by Manila (Admin only)."""
    driver_options = _extract_key_value_options(args, 'driver_options')
    manage_kwargs = {'driver_options': driver_options}
    if cs.api_version < api_versions.APIVersion('2.51'):
        if getattr(args, 'share_network_subnet'):
            raise exceptions.CommandError('Share network subnet option is only available with manila API version >= 2.51')
    else:
        manage_kwargs['share_network_subnet_id'] = args.share_network_subnet
    share_server = cs.share_servers.manage(args.host, args.share_network, args.identifier, **manage_kwargs)
    if args.wait:
        try:
            _wait_for_resource_status(cs, share_server, resource_type='share_server', expected_status='active')
        except exceptions.CommandError as e:
            print(e, file=sys.stderr)
    cliutils.print_dict(share_server._info)