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
def convert_string_to_case_insensitive(data):
    """Convert a string value into a lower case string.

    This effectively makes the string case-insensitive.

    :param data: The value to convert.
    :return: The lower-cased string representation of the value, or None is
        'data' is None.
    :raises InvalidInput: If the value is not a string.
    """
    try:
        return data.lower()
    except AttributeError as e:
        error_message = _('Input value %s must be string type') % data
        raise n_exc.InvalidInput(error_message=error_message) from e