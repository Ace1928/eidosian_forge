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
@property
def max_command_size_limit(self):
    try:
        return max(self.command_size_limits.values())
    except ValueError:
        return self.command_size_limit