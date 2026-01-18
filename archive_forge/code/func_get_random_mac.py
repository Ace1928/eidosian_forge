import random
import socket
import netaddr
from neutron_lib import constants
def get_random_mac(base_mac):
    """Get a random MAC address string of the specified base format.

    The first 3 octets will remain unchanged. If the 4th octet is not
    00, it will also be used. The others will be randomly generated.

    :param base_mac: Base MAC address represented by an array of 6 strings/int
    :returns: The MAC address string.
    """
    mac = [int(base_mac[0], 16), int(base_mac[1], 16), int(base_mac[2], 16), random.getrandbits(8), random.getrandbits(8), random.getrandbits(8)]
    if base_mac[3] != '00':
        mac[3] = int(base_mac[3], 16)
    return ':'.join(['%02x' % x for x in mac])