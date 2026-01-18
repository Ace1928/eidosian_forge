import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from urllib3.util import make_headers
from urllib3.util import parse_url
from .__version__ import __version__
from . import certs
from ._internal_utils import to_native_string
from .compat import parse_http_list as _parse_list_header
from .compat import (
from .cookies import cookiejar_from_dict
from .structures import CaseInsensitiveDict
from .exceptions import (
def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """
    get_proxy = lambda k: os.environ.get(k) or os.environ.get(k.upper())
    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy('no_proxy')
    parsed = urlparse(url)
    if parsed.hostname is None:
        return True
    if no_proxy:
        no_proxy = (host for host in no_proxy.replace(' ', '').split(',') if host)
        if is_ipv4_address(parsed.hostname):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed.hostname, proxy_ip):
                        return True
                elif parsed.hostname == proxy_ip:
                    return True
        else:
            host_with_port = parsed.hostname
            if parsed.port:
                host_with_port += ':{}'.format(parsed.port)
            for host in no_proxy:
                if parsed.hostname.endswith(host) or host_with_port.endswith(host):
                    return True
    with set_environ('no_proxy', no_proxy_arg):
        try:
            bypass = proxy_bypass(parsed.hostname)
        except (TypeError, socket.gaierror):
            bypass = False
    if bypass:
        return True
    return False