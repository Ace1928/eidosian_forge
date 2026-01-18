import re
from string import ascii_letters, digits, hexdigits
def header_check(octet):
    """Return True if the octet should be escaped with header quopri."""
    return chr(octet) != _QUOPRI_HEADER_MAP[octet]