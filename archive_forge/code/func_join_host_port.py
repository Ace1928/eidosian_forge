import socket
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union
from selenium.types import AnyKey
from selenium.webdriver.common.keys import Keys
def join_host_port(host: str, port: int) -> str:
    """Joins a hostname and port together.

    This is a minimal implementation intended to cope with IPv6 literals. For
    example, _join_host_port('::1', 80) == '[::1]:80'.

    :Args:
        - host - A hostname.
        - port - An integer port.
    """
    if ':' in host and (not host.startswith('[')):
        return f'[{host}]:{port}'
    return f'{host}:{port}'