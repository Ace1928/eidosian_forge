from __future__ import (absolute_import, division, print_function)
import os
import hmac
import re
from ansible.module_utils.six.moves.urllib.parse import urlparse
def is_ssh_url(url):
    """ check if url is ssh """
    if '@' in url and '://' not in url:
        return True
    for scheme in ('ssh://', 'git+ssh://', 'ssh+git://'):
        if url.startswith(scheme):
            return True
    return False