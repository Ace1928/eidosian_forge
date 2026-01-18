import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def format_binary_line(data):
    left = ' '.join(['{:02X}'.format(byte_ord(c)) for c in data])
    right = ''.join(['.{:c}..'.format(byte_ord(c))[(byte_ord(c) + 63) // 95] for c in data])
    return '{:50s} {}'.format(left, right)