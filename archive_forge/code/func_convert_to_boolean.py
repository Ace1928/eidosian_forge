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
def convert_to_boolean(data):
    """Convert a data value into a python bool.

    :param data: The data value to convert to a python bool. This function
        supports string types, bools, and ints for conversion of representation
        to python bool.
    :returns: The bool value of 'data' if it can be coerced.
    :raises InvalidInput: If the value can't be coerced to a python bool.
    """
    try:
        return strutils.bool_from_string(data, strict=True)
    except ValueError as e:
        msg = _("'%s' cannot be converted to boolean") % data
        raise n_exc.InvalidInput(error_message=msg) from e