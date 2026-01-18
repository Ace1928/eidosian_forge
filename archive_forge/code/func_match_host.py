from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence
from tornado import netutil
def match_host(host: str, pattern: str) -> bool:
    """ Match a host string against a pattern

    Args:
        host (str)
            A hostname to compare to the given pattern

        pattern (str)
            A string representing a hostname pattern, possibly including
            wildcards for ip address octets or ports.

    This function will return ``True`` if the hostname matches the pattern,
    including any wildcards. If the pattern contains a port, the host string
    must also contain a matching port.

    Returns:
        bool

    Examples:

        >>> match_host('192.168.0.1:80', '192.168.0.1:80')
        True
        >>> match_host('192.168.0.1:80', '192.168.0.1')
        True
        >>> match_host('192.168.0.1:80', '192.168.0.1:8080')
        False
        >>> match_host('192.168.0.1', '192.168.0.2')
        False
        >>> match_host('192.168.0.1', '192.168.*.*')
        True
        >>> match_host('alice', 'alice')
        True
        >>> match_host('alice:80', 'alice')
        True
        >>> match_host('alice', 'bob')
        False
        >>> match_host('foo.example.com', 'foo.example.com.net')
        False
        >>> match_host('alice', '*')
        True
        >>> match_host('alice', '*:*')
        True
        >>> match_host('alice:80', '*')
        True
        >>> match_host('alice:80', '*:80')
        True
        >>> match_host('alice:8080', '*:80')
        False

    """
    host_port: str | None = None
    if ':' in host:
        host, host_port = host.rsplit(':', 1)
    pattern_port: str | None = None
    if ':' in pattern:
        pattern, pattern_port = pattern.rsplit(':', 1)
        if pattern_port == '*':
            pattern_port = None
    if pattern_port is not None and host_port != pattern_port:
        return False
    host_parts = host.split('.')
    pattern_parts = pattern.split('.')
    if len(pattern_parts) > len(host_parts):
        return False
    for h, p in zip(host_parts, pattern_parts):
        if h == p or p == '*':
            continue
        else:
            return False
    return True