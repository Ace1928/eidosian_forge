import argparse
import logging
import os
import sys
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import strutils
from magnumclient.common import cliutils
from magnumclient import exceptions as exc
from magnumclient.i18n import _
from magnumclient.v1 import client as client_v1
from magnumclient.v1 import shell as shell_v1
from magnumclient import version
def get_base_parser(self):
    parser = MagnumClientArgumentParser(prog='magnum', description=__doc__.strip(), epilog='See "magnum help COMMAND" for help on a specific command.', add_help=False, formatter_class=OpenStackHelpFormatter)
    parser.add_argument('-h', '--help', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--version', action='version', version=version.version_info.version_string())
    parser.add_argument('--debug', default=False, action='store_true', help=_('Print debugging output.'))
    parser.add_argument('--os-cache', default=strutils.bool_from_string(cliutils.env('OS_CACHE', default=False)), action='store_true', help=_('Use the auth token cache. Defaults to False if env[OS_CACHE] is not set.'))
    parser.add_argument('--os-region-name', metavar='<region-name>', default=os.environ.get('OS_REGION_NAME'), help=_('Region name. Default=env[OS_REGION_NAME].'))
    parser.add_argument('--os-auth-url', metavar='<auth-auth-url>', default=cliutils.env('OS_AUTH_URL', default=None), help=_('Defaults to env[OS_AUTH_URL].'))
    parser.add_argument('--os-user-id', metavar='<auth-user-id>', default=cliutils.env('OS_USER_ID', default=None), help=_('Defaults to env[OS_USER_ID].'))
    parser.add_argument('--os-username', metavar='<auth-username>', default=cliutils.env('OS_USERNAME', default=None), help=_('Defaults to env[OS_USERNAME].'))
    parser.add_argument('--os-user-domain-id', metavar='<auth-user-domain-id>', default=cliutils.env('OS_USER_DOMAIN_ID', default=None), help=_('Defaults to env[OS_USER_DOMAIN_ID].'))
    parser.add_argument('--os-user-domain-name', metavar='<auth-user-domain-name>', default=cliutils.env('OS_USER_DOMAIN_NAME', default=None), help=_('Defaults to env[OS_USER_DOMAIN_NAME].'))
    parser.add_argument('--os-project-id', metavar='<auth-project-id>', default=cliutils.env('OS_PROJECT_ID', default=None), help=_('Defaults to env[OS_PROJECT_ID].'))
    parser.add_argument('--os-project-name', metavar='<auth-project-name>', default=cliutils.env('OS_PROJECT_NAME', default=None), help=_('Defaults to env[OS_PROJECT_NAME].'))
    parser.add_argument('--os-tenant-id', metavar='<auth-tenant-id>', default=cliutils.env('OS_TENANT_ID', default=None), help=argparse.SUPPRESS)
    parser.add_argument('--os-tenant-name', metavar='<auth-tenant-name>', default=cliutils.env('OS_TENANT_NAME', default=None), help=argparse.SUPPRESS)
    parser.add_argument('--os-project-domain-id', metavar='<auth-project-domain-id>', default=cliutils.env('OS_PROJECT_DOMAIN_ID', default=None), help=_('Defaults to env[OS_PROJECT_DOMAIN_ID].'))
    parser.add_argument('--os-project-domain-name', metavar='<auth-project-domain-name>', default=cliutils.env('OS_PROJECT_DOMAIN_NAME', default=None), help=_('Defaults to env[OS_PROJECT_DOMAIN_NAME].'))
    parser.add_argument('--os-token', metavar='<auth-token>', default=cliutils.env('OS_TOKEN', default=None), help=_('Defaults to env[OS_TOKEN].'))
    parser.add_argument('--os-password', metavar='<auth-password>', default=cliutils.env('OS_PASSWORD', default=None), help=_('Defaults to env[OS_PASSWORD].'))
    parser.add_argument('--service-type', metavar='<service-type>', help=_('Defaults to container-infra for all actions.'))
    parser.add_argument('--service_type', help=argparse.SUPPRESS)
    parser.add_argument('--endpoint-type', metavar='<endpoint-type>', default=cliutils.env('OS_ENDPOINT_TYPE', default=None), help=argparse.SUPPRESS)
    parser.add_argument('--os-endpoint-type', metavar='<os-endpoint-type>', default=cliutils.env('OS_ENDPOINT_TYPE', default=None), help=_('Defaults to env[OS_ENDPOINT_TYPE]'))
    parser.add_argument('--os-interface', metavar='<os-interface>', default=cliutils.env('OS_INTERFACE', default=DEFAULT_INTERFACE), help=argparse.SUPPRESS)
    parser.add_argument('--os-cloud', metavar='<auth-cloud>', default=cliutils.env('OS_CLOUD', default=None), help=_('Defaults to env[OS_CLOUD].'))
    parser.add_argument('--magnum-api-version', metavar='<magnum-api-ver>', default=cliutils.env('MAGNUM_API_VERSION', default='latest'), help=_('Accepts "api", defaults to env[MAGNUM_API_VERSION].'))
    parser.add_argument('--magnum_api_version', help=argparse.SUPPRESS)
    parser.add_argument('--os-cacert', metavar='<ca-certificate>', default=cliutils.env('OS_CACERT', default=None), help=_('Specify a CA bundle file to use in verifying a TLS (https) server certificate. Defaults to env[OS_CACERT].'))
    parser.add_argument('--os-endpoint-override', metavar='<endpoint-override>', default=cliutils.env('OS_ENDPOINT_OVERRIDE', default=None), help=_('Use this API endpoint instead of the Service Catalog.'))
    parser.add_argument('--bypass-url', metavar='<bypass-url>', default=cliutils.env('BYPASS_URL', default=None), dest='bypass_url', help=argparse.SUPPRESS)
    parser.add_argument('--bypass_url', help=argparse.SUPPRESS)
    parser.add_argument('--insecure', default=cliutils.env('MAGNUMCLIENT_INSECURE', default=False), action='store_true', help=_('Do not verify https connections'))
    if profiler:
        parser.add_argument('--profile', metavar='HMAC_KEY', default=cliutils.env('OS_PROFILE', default=None), help='HMAC key to use for encrypting context data for performance profiling of operation. This key should be the value of the HMAC key configured for the OSprofiler middleware in magnum; it is specified in the Magnum configuration file at "/etc/magnum/magnum.conf". Without the key, profiling will not be triggered even if OSprofiler is enabled on the server side.')
    return parser