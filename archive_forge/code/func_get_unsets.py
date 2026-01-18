import functools
import ipaddress
from openstackclient.identity import common as identity_common
from osc_lib import exceptions as osc_exc
from osc_lib import utils
from oslo_utils import uuidutils
from octaviaclient.api import exceptions
from octaviaclient.osc.v2 import constants
def get_unsets(parsed_args):
    unsets = {}
    for arg, value in vars(parsed_args).items():
        if value and arg == 'tags':
            unsets[arg] = value
        elif value is True and arg not in ('wait', 'all_tag'):
            unsets[arg] = None
    return unsets