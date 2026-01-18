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
def convert_to_protocol(data):
    """Validate that a specified IP protocol is valid.

    For the authoritative list mapping protocol names to numbers, see the IANA:
    http://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml

    :param data: The value to verify is an IP protocol.
    :returns: If data is an int between 0 and 255 or None, return that; if
        data is a string then return it lower-cased if it matches one of the
        allowed protocol names.
    :raises exceptions.InvalidInput: If data is an int < 0, an
        int > 255, or a string that does not match one of the allowed protocol
        names.
    """
    if data is None:
        return
    val = convert_string_to_case_insensitive(data)
    if val in constants.IPTABLES_PROTOCOL_MAP:
        return data
    error_message = _("IP protocol '%s' is not supported. Only protocol names and their integer representation (0 to 255) are supported") % data
    try:
        if validators.validate_range(convert_to_int(data), [0, 255]) is None:
            return data
        else:
            raise n_exc.InvalidInput(error_message=error_message)
    except n_exc.InvalidInput as e:
        raise n_exc.InvalidInput(error_message=error_message) from e