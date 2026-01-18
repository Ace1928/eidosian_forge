from __future__ import annotations
import re
from ipaddress import AddressValueError, IPv6Address
from urllib.parse import scheme_chars
def looks_like_ip(maybe_ip: str) -> bool:
    """Check whether the given str looks like an IPv4 address."""
    if not maybe_ip[0].isdigit():
        return False
    return IP_RE.fullmatch(maybe_ip) is not None