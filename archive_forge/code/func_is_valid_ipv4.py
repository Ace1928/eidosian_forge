import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_ipv4(ipv4):
    """Returns True if given is a valid ipv4 address.

    Given value should be a dot-decimal notation string.

    Samples:
        - valid address: 10.0.0.1, 192.168.0.1
        - invalid address: 11.0.0, 192:168:0:1, etc.
    """
    return ip.valid_ipv4(ipv4)