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
def convert_ip_to_canonical_format(value):
    """IP Address is validated and then converted to canonical format.

    :param value: The IP Address which needs to be checked.
    :returns: - None if 'value' is None,
              - 'value' if 'value' is IPv4 address,
              - 'value' if 'value' is not an IP Address
              - canonical IPv6 address if 'value' is IPv6 address.

    """
    try:
        ip = netaddr.IPAddress(value)
        if ip.version == constants.IP_VERSION_6:
            return str(ip.format(dialect=netaddr.ipv6_compact))
    except (netaddr.core.AddrFormatError, ValueError):
        pass
    return value