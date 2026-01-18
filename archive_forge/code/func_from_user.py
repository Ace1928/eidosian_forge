import abc
import base64
import collections
import copy
import functools
import io
import itertools
import math
import operator
import re
import socket
import struct
import netaddr
from os_ken.lib.stringify import StringifyMixin
from os_ken.lib.packet import afi as addr_family
from os_ken.lib.packet import safi as subaddr_family
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet import vxlan
from os_ken.lib.packet import mpls
from os_ken.lib import addrconv
from os_ken.lib import type_desc
from os_ken.lib.type_desc import TypeDisp
from os_ken.lib import ip
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.utils import binary_str
from os_ken.utils import import_module
@classmethod
def from_user(cls, route_dist, **kwargs):
    """
        Utility method for creating a L2VPN NLRI instance.

        This function returns a L2VPN NLRI instance
        from human readable format value.

        :param kwargs: The following arguments are available.

        ============== ============= ========= ==============================
        Argument       Value         Operator  Description
        ============== ============= ========= ==============================
        ether_type     Integer       Numeric   Ethernet Type.
        src_mac        Mac Address   Nothing   Source Mac address.
        dst_mac        Mac Address   Nothing   Destination Mac address.
        llc_ssap       Integer       Numeric   Source Service Access Point
                                               in LLC.
        llc_dsap       Integer       Numeric   Destination Service Access
                                               Point in LLC.
        llc_control    Integer       Numeric   Control field in LLC.
        snap           Integer       Numeric   Sub-Network Access Protocol
                                               field.
        vlan_id        Integer       Numeric   VLAN ID.
        vlan_cos       Integer       Numeric   VLAN COS field.
        inner_vlan_id  Integer       Numeric   Inner VLAN ID.
        inner_vlan_cos Integer       Numeric   Inner VLAN COS field.
        ============== ============= ========= ==============================
        """
    return cls._from_user(route_dist, **kwargs)