import re
from string import ascii_letters, digits, hexdigits
def body_check(octet):
    """Return True if the octet should be escaped with body quopri."""
    return chr(octet) != _QUOPRI_BODY_MAP[octet]