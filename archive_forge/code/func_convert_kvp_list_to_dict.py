import copy
import uuid
import netaddr
from oslo_config import cfg
from oslo_utils import strutils
from neutron_lib._i18n import _
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.placement import utils as pl_utils
from neutron_lib.utils import net as net_utils
def convert_kvp_list_to_dict(kvp_list):
    """Convert a list of 'key=value' strings to a dict.

    :param kvp_list: A list of key value pair strings. For more info on the
        format see; convert_kvp_str_to_list().
    :returns: A dict who's key value pairs are populated by parsing 'kvp_list'.
    :raises InvalidInput: If any of the key value strings are malformed.
    """
    if kvp_list == ['True']:
        return {}
    kvp_map = {}
    for kvp_str in kvp_list:
        key, value = convert_kvp_str_to_list(kvp_str)
        kvp_map.setdefault(key, set())
        kvp_map[key].add(value)
    return dict(((x, list(y)) for x, y in kvp_map.items()))