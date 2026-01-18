import numbers
import re
import socket
from os_ken.lib import ip
def is_valid_old_asn(asn):
    """Returns True if the given AS number is Two Octet."""
    return isinstance(asn, numbers.Integral) and 0 <= asn <= 65535