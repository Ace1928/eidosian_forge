import struct
from os_ken.lib import addrconv
from . import ether_types as ether
from . import packet_base
def arp_ip(opcode, src_mac, src_ip, dst_mac, dst_ip):
    """A convenient wrapper for IPv4 ARP for Ethernet.

    This is an equivalent of the following code.

        arp(ARP_HW_TYPE_ETHERNET, ether.ETH_TYPE_IP,                6, 4, opcode, src_mac, src_ip, dst_mac, dst_ip)
    """
    return arp(ARP_HW_TYPE_ETHERNET, ether.ETH_TYPE_IP, 6, 4, opcode, src_mac, src_ip, dst_mac, dst_ip)