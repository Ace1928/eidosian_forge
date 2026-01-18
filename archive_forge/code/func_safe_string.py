import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def safe_string(s):
    out = b''
    for c in s:
        i = byte_ord(c)
        if 32 <= i <= 127:
            out += byte_chr(i)
        else:
            out += b('%{:02X}'.format(i))
    return out