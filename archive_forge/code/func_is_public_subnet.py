import socket
import struct
def is_public_subnet(ip):
    """
    Utility function to check if an IP address is inside a public subnet.

    :type ip: ``str``
    :param ip: IP address to check

    :return: ``bool`` if the specified IP address is public.
    """
    return not is_private_subnet(ip=ip)