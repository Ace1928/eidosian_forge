import os
import socket
import struct
from os_ken import cfg
from os_ken.base.app_manager import OSKenApp
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet import zebra
from os_ken.lib.packet import safi as packet_safi
from os_ken.services.protocols.zebra import event
from os_ken.services.protocols.zebra.client import event as zclient_event
def get_zebra_route_type_by_name(route_type='BGP'):
    """
    Returns the constant value for Zebra route type named "ZEBRA_ROUTE_*"
    from its name.

    See "ZEBRA_ROUTE_*" constants in "os_ken.lib.packet.zebra" module.

    :param route_type: Route type name (e.g., Kernel, BGP).
    :return: Constant value for Zebra route type.
    """
    return getattr(zebra, 'ZEBRA_ROUTE_%s' % route_type.upper())