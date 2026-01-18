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
@cliutils.arg('type', metavar='<type>', help="Security service type: 'ldap', 'kerberos' or 'active_directory'.")
@cliutils.arg('--dns-ip', metavar='<dns_ip>', default=None, help="DNS IP address used inside project's network.")
@cliutils.arg('--ou', metavar='<ou>', default=None, help='Security service OU (Organizational Unit). Available only for microversion >= 2.44.')
@cliutils.arg('--server', metavar='<server>', default=None, help='Security service IP address or hostname.')
@cliutils.arg('--domain', metavar='<domain>', default=None, help='Security service domain.')
@cliutils.arg('--user', metavar='<user>', default=None, help='Security service user or group used by project.')
@cliutils.arg('--password', metavar='<password>', default=None, help='Password used by user.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Security service name.')
@cliutils.arg('--default-ad-site', metavar='<default_ad_site>', dest='default_ad_site', default=None, help="Default AD site. Available only for microversion >= 2.76. Can be provided in the place of '--server' but not along with it.")
@cliutils.arg('--description', metavar='<description>', default=None, help='Security service description.')
def do_security_service_create(cs, args):
    """Create security service used by project."""
    values = {'dns_ip': args.dns_ip, 'server': args.server, 'domain': args.domain, 'user': args.user, 'password': args.password, 'name': args.name, 'description': args.description}
    if cs.api_version.matches(api_versions.APIVersion('2.44'), api_versions.APIVersion()):
        values['ou'] = args.ou
    elif args.ou:
        raise exceptions.CommandError('Security service Organizational Unit (ou) option is only available with manila API version >= 2.44')
    if cs.api_version.matches(api_versions.APIVersion('2.76'), api_versions.APIVersion()):
        values['default_ad_site'] = args.default_ad_site
    elif args.default_ad_site:
        raise exceptions.CommandError('Default AD site option is only available with manila API version >= 2.76')
    if args.type == 'active_directory':
        if args.server and args.default_ad_site:
            raise exceptions.CommandError("Cannot create security service because both server and 'default_ad_site' were provided. Specify either server or 'default_ad_site'.")
    security_service = cs.security_services.create(args.type, **values)
    info = security_service._info.copy()
    cliutils.print_dict(info)