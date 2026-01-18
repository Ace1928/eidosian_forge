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
def convert_to_positive_float_or_none(val):
    """Converts a value to a python float if the value is positive.

    :param val: The value to convert to a positive python float.
    :returns: The value as a python float. If the val is None, None is
        returned.
    :raises ValueError, InvalidInput: A ValueError is raised if the 'val'
        is a float, but is negative. InvalidInput is raised if 'val' can't be
        converted to a python float.
    """
    if val is None:
        return
    try:
        val = float(val)
        if val < 0:
            raise ValueError()
    except (ValueError, TypeError) as e:
        msg = _("'%s' must be a non negative decimal") % val
        raise n_exc.InvalidInput(error_message=msg) from e
    return val