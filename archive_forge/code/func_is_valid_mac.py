import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_mac(mac):
    """Returns True if the given MAC address is valid.

    The given MAC address should be a colon hexadecimal notation string.

    Samples:
        - valid address: aa:bb:cc:dd:ee:ff, 11:22:33:44:55:66
        - invalid address: aa:bb:cc:dd, 11-22-33-44-55-66, etc.
    """
    return bool(re.match('^' + '[\\:\\-]'.join(['([0-9a-f]{2})'] * 6) + '$', mac.lower()))