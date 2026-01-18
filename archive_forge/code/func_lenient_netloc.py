from __future__ import annotations
import re
from ipaddress import AddressValueError, IPv6Address
from urllib.parse import scheme_chars
def lenient_netloc(url: str) -> str:
    """Extract the netloc of a URL-like string.

    Similar to the netloc attribute returned by
    urllib.parse.{urlparse,urlsplit}, but extract more leniently, without
    raising errors.
    """
    after_userinfo = _schemeless_url(url).partition('/')[0].partition('?')[0].partition('#')[0].rpartition('@')[-1]
    if after_userinfo and after_userinfo[0] == '[':
        maybe_ipv6 = after_userinfo.partition(']')
        if maybe_ipv6[1] == ']':
            return f'{maybe_ipv6[0]}]'
    hostname = after_userinfo.partition(':')[0].strip()
    without_root_label = hostname.rstrip('.。．｡')
    return without_root_label