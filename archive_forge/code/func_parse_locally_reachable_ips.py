from __future__ import (absolute_import, division, print_function)
import glob
import os
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network, NetworkCollector
from ansible.module_utils.facts.utils import get_file_content
def parse_locally_reachable_ips(output):
    for line in output.splitlines():
        if not line:
            continue
        words = line.split()
        if words[0] != 'local':
            continue
        address = words[1]
        if ':' in address:
            if address not in locally_reachable_ips['ipv6']:
                locally_reachable_ips['ipv6'].append(address)
        elif address not in locally_reachable_ips['ipv4']:
            locally_reachable_ips['ipv4'].append(address)