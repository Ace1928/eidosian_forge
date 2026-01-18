import logging
from osc_lib import utils
def build_option_parser(parser):
    """Hook to add global options"""
    parser.add_argument('--os-container-infra-api-version', metavar='<container-infra-api-version>', default=utils.env('OS_CONTAINER_INFRA_API_VERSION', default=DEFAULT_MAJOR_API_VERSION), help='Container-Infra API version, default=' + DEFAULT_MAJOR_API_VERSION + ' (Env: OS_CONTAINER_INFRA_API_VERSION)')
    return parser