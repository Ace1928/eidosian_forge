import sys
import os
import errno
import getopt
import time
import socket
import collections
from warnings import _deprecated, warn
from email._header_value_parser import get_addr_spec, get_angle_addr
import asyncore
import asynchat
def collect_incoming_data(self, data):
    limit = None
    if self.smtp_state == self.COMMAND:
        limit = self.max_command_size_limit
    elif self.smtp_state == self.DATA:
        limit = self.data_size_limit
    if limit and self.num_bytes > limit:
        return
    elif limit:
        self.num_bytes += len(data)
    if self._decode_data:
        self.received_lines.append(str(data, 'utf-8'))
    else:
        self.received_lines.append(data)