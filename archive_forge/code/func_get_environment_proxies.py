from __future__ import annotations
import codecs
import email.message
import ipaddress
import mimetypes
import os
import re
import time
import typing
from pathlib import Path
from urllib.request import getproxies
import sniffio
from ._types import PrimitiveData
def get_environment_proxies() -> dict[str, str | None]:
    """Gets proxy information from the environment"""
    proxy_info = getproxies()
    mounts: dict[str, str | None] = {}
    for scheme in ('http', 'https', 'all'):
        if proxy_info.get(scheme):
            hostname = proxy_info[scheme]
            mounts[f'{scheme}://'] = hostname if '://' in hostname else f'http://{hostname}'
    no_proxy_hosts = [host.strip() for host in proxy_info.get('no', '').split(',')]
    for hostname in no_proxy_hosts:
        if hostname == '*':
            return {}
        elif hostname:
            if '://' in hostname:
                mounts[hostname] = None
            elif is_ipv4_hostname(hostname):
                mounts[f'all://{hostname}'] = None
            elif is_ipv6_hostname(hostname):
                mounts[f'all://[{hostname}]'] = None
            elif hostname.lower() == 'localhost':
                mounts[f'all://{hostname}'] = None
            else:
                mounts[f'all://*{hostname}'] = None
    return mounts