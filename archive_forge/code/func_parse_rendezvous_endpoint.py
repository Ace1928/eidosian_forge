import ipaddress
import random
import re
import socket
import time
import weakref
from datetime import timedelta
from threading import Event, Thread
from typing import Any, Callable, Dict, Optional, Tuple, Union
def parse_rendezvous_endpoint(endpoint: Optional[str], default_port: int) -> Tuple[str, int]:
    """Extract the hostname and the port number from a rendezvous endpoint.

    Args:
        endpoint:
            A string in format <hostname>[:<port>].
        default_port:
            The port number to use if the endpoint does not include one.

    Returns:
        A tuple of hostname and port number.
    """
    if endpoint is not None:
        endpoint = endpoint.strip()
    if not endpoint:
        return ('localhost', default_port)
    if endpoint[0] == '[' and endpoint[-1] == ']':
        host, *rest = (endpoint, *[])
    else:
        host, *rest = endpoint.rsplit(':', 1)
    if len(host) > 1 and host[0] == '[' and (host[-1] == ']'):
        host = host[1:-1]
    if len(rest) == 1:
        port = _try_parse_port(rest[0])
        if port is None or port >= 2 ** 16:
            raise ValueError(f"The port number of the rendezvous endpoint '{endpoint}' must be an integer between 0 and 65536.")
    else:
        port = default_port
    if not re.match('^[\\w\\.:-]+$', host):
        raise ValueError(f"The hostname of the rendezvous endpoint '{endpoint}' must be a dot-separated list of labels, an IPv4 address, or an IPv6 address.")
    return (host, port)