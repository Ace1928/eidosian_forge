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
def convert_to_int_if_not_none(data):
    """Uses convert_to_int() on the data if the data is not None.

    :param data: The data value to convert.
    :returns: The 'data' returned from convert_to_int() if 'data' is not None.
        None is returned if data is None.
    """
    if data is not None:
        return convert_to_int(data)
    return data