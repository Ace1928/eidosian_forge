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
@cliutils.arg('share', metavar='<share>', help='Name or ID of the NAS share to modify.')
@cliutils.arg('access_type', metavar='<access_type>', help='Access rule type (only "ip", "user"(user or group), "cert" or "cephx" are supported).')
@cliutils.arg('access_to', metavar='<access_to>', help='Value that defines access.')
@cliutils.arg('--access-level', '--access_level', metavar='<access_level>', type=str, default=None, choices=['rw', 'ro'], action='single_alias', help='Share access level ("rw" and "ro" access levels are supported). Defaults to rw.')
@cliutils.arg('--metadata', type=str, nargs='*', metavar='<key=value>', help='Space Separated list of key=value pairs of metadata items. OPTIONAL: Default=None. Available only for microversion >= 2.45.', default=None)
@cliutils.arg('--wait', action='store_true', help='Wait for share access to become active')
def do_access_allow(cs, args):
    """Allow access to a given share."""
    access_metadata = None
    if cs.api_version.matches(api_versions.APIVersion('2.45'), api_versions.APIVersion()):
        access_metadata = _extract_metadata(args)
    elif getattr(args, 'metadata'):
        raise exceptions.CommandError('Adding metadata to access rules is supported only beyond API version 2.45')
    share = _find_share(cs, args.share)
    access = share.allow(args.access_type, args.access_to, args.access_level, access_metadata)
    if args.wait:
        try:
            if not cs.api_version.matches(api_versions.APIVersion('2.45'), api_versions.APIVersion()):
                raise exceptions.CommandError('Waiting on the allowing access operation is only available for API versions equal to or greater than 2.45.')
            access_id = access.get('id')
            share_access_rule = cs.share_access_rules.get(access_id)
            access = _wait_for_resource_status(cs, share_access_rule, resource_type='share_access_rule', expected_status='active', status_attr='state')._info
        except exceptions.CommandError as e:
            print(e, file=sys.stderr)
    cliutils.print_dict(access)