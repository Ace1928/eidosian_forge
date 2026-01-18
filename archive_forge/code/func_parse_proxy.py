from __future__ import unicode_literals
import sys
import copy
import hashlib
import logging
import os
import tempfile
import warnings
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, REVERSE_TYPE_MAP, Struct
from .transport import get_http_wrapper, set_http_wrapper, get_Http
from .helpers import Alias, fetch, sort_dict, make_key, process_element, \
from .wsse import UsernameToken
def parse_proxy(proxy_str):
    """Parses proxy address user:pass@host:port into a dict suitable for httplib2"""
    proxy_dict = {}
    if proxy_str is None:
        return
    if '@' in proxy_str:
        user_pass, host_port = proxy_str.split('@')
    else:
        user_pass, host_port = ('', proxy_str)
    if ':' in host_port:
        host, port = host_port.split(':')
        proxy_dict['proxy_host'], proxy_dict['proxy_port'] = (host, int(port))
    if ':' in user_pass:
        proxy_dict['proxy_user'], proxy_dict['proxy_pass'] = user_pass.split(':')
    return proxy_dict