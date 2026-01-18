import struct
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ether_types as ether
from os_ken.lib.packet import in_proto as inet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import ipv6
from os_ken.lib.packet import packet
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import packet_utils
from os_ken.lib.packet import vlan
from os_ken.lib import addrconv
@staticmethod
def create_version(version, type_, vrid, priority, max_adver_int, ip_addresses, auth_type=None, auth_data=None):
    cls_ = vrrp._VRRP_VERSIONS.get(version, None)
    if not cls_:
        raise ValueError('unknown VRRP version %d' % version)
    if priority is None:
        priority = VRRP_PRIORITY_BACKUP_DEFAULT
    count_ip = len(ip_addresses)
    if max_adver_int is None:
        max_adver_int = cls_.sec_to_max_adver_int(VRRP_MAX_ADVER_INT_DEFAULT_IN_SEC)
    return cls_(version, type_, vrid, priority, count_ip, max_adver_int, None, ip_addresses, auth_type=auth_type, auth_data=auth_data)