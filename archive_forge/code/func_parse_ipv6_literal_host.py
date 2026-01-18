from __future__ import annotations
import re
import sys
import warnings
from typing import (
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import (
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
def parse_ipv6_literal_host(entity: str, default_port: Optional[int]) -> tuple[str, Optional[Union[str, int]]]:
    """Validates an IPv6 literal host:port string.

    Returns a 2-tuple of IPv6 literal followed by port where
    port is default_port if it wasn't specified in entity.

    :Parameters:
        - `entity`: A string that represents an IPv6 literal enclosed
                    in braces (e.g. '[::1]' or '[::1]:27017').
        - `default_port`: The port number to use when one wasn't
                          specified in entity.
    """
    if entity.find(']') == -1:
        raise ValueError("an IPv6 address literal must be enclosed in '[' and ']' according to RFC 2732.")
    i = entity.find(']:')
    if i == -1:
        return (entity[1:-1], default_port)
    return (entity[1:i], entity[i + 2:])