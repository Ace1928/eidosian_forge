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
def convert_none_to_empty_string(value):
    """Convert the value to an empty string if it's None.

    :param value: The value to convert.
    :returns: An empty string if 'value' is None, otherwise 'value'.
    """
    return '' if value is None else value