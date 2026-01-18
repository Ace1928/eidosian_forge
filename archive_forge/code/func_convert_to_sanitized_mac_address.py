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
def convert_to_sanitized_mac_address(mac_address):
    """Return a MAC address with format xx:xx:xx:xx:xx:xx

    :param mac_address: (string, netaddr.EUI) The MAC address value
    :return: A string with the MAC address formatted. If the MAC address
             provided is invalid, the same input value is returned; the goal
             of this method is not to validate it.
    """
    try:
        if isinstance(mac_address, netaddr.EUI):
            _mac_address = copy.deepcopy(mac_address)
            _mac_address.dialect = netaddr.mac_unix_expanded
            return str(_mac_address)
        return str(netaddr.EUI(mac_address, dialect=netaddr.mac_unix_expanded))
    except netaddr.core.AddrFormatError:
        return mac_address