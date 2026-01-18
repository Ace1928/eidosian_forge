import socket
import struct
def is_private_subnet(ip):
    """
    Utility function to check if an IP address is inside a private subnet.

    :type ip: ``str``
    :param ip: IP address to check

    :return: ``bool`` if the specified IP address is private.
    """
    priv_subnets = [{'subnet': '10.0.0.0', 'mask': '255.0.0.0'}, {'subnet': '172.16.0.0', 'mask': '255.240.0.0'}, {'subnet': '192.168.0.0', 'mask': '255.255.0.0'}]
    ip = struct.unpack('I', socket.inet_aton(ip))[0]
    for network in priv_subnets:
        subnet = struct.unpack('I', socket.inet_aton(network['subnet']))[0]
        mask = struct.unpack('I', socket.inet_aton(network['mask']))[0]
        if ip & mask == subnet & mask:
            return True
    return False