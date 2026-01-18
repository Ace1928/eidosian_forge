from __future__ import (absolute_import, division, print_function)
import re
import socket
import struct
from ansible.module_utils.facts.network.base import Network
def parse_status_line(self, words, current_if, ips):
    current_if['status'] = words[1]