from __future__ import (absolute_import, division, print_function)
import os
import hmac
import re
from ansible.module_utils.six.moves.urllib.parse import urlparse
def get_fqdn_and_port(repo_url):
    """ chop the hostname and port out of a url """
    fqdn = None
    port = None
    ipv6_re = re.compile('(\\[[^]]*\\])(?::([0-9]+))?')
    if '@' in repo_url and '://' not in repo_url:
        repo_url = repo_url.split('@', 1)[1]
        match = ipv6_re.match(repo_url)
        if match:
            fqdn, path = match.groups()
        elif ':' in repo_url:
            fqdn = repo_url.split(':')[0]
        elif '/' in repo_url:
            fqdn = repo_url.split('/')[0]
    elif '://' in repo_url:
        parts = urlparse(repo_url)
        if parts[1] != '':
            fqdn = parts[1]
            if '@' in fqdn:
                fqdn = fqdn.split('@', 1)[1]
            match = ipv6_re.match(fqdn)
            if match:
                fqdn, port = match.groups()
            elif ':' in fqdn:
                fqdn, port = fqdn.split(':')[0:2]
    return (fqdn, port)