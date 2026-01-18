from __future__ import annotations
import re
from ipaddress import AddressValueError, IPv6Address
from urllib.parse import scheme_chars
def looks_like_ipv6(maybe_ip: str) -> bool:
    """Check whether the given str looks like an IPv6 address."""
    try:
        IPv6Address(maybe_ip)
    except AddressValueError:
        return False
    return True