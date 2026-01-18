from netaddr.core import AddrFormatError
from netaddr.ip import IPAddress
def chr_range(low, high):
    """Returns all characters between low and high chars."""
    return [chr(i) for i in range(ord(low), ord(high) + 1)]