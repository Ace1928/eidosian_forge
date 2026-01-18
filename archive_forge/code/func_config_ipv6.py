from __future__ import (absolute_import, division, print_function)
import time
from datetime import datetime
import re
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
def config_ipv6(hostname):
    ip_addr, port = (hostname, None)
    if hostname.count(':') == 1:
        ip_addr, port = hostname.split(':')
    if not re.match(HOSTNAME_REGEX, ip_addr):
        if ']:' in ip_addr:
            ip_addr, port = ip_addr.split(']:')
        ip_addr = ip_addr.strip('[]')
        if port is None or port == '':
            hostname = '[{0}]'.format(ip_addr)
        else:
            hostname = '[{0}]:{1}'.format(ip_addr, port)
    return hostname