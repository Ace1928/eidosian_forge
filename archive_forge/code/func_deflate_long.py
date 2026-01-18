import sys
import struct
import traceback
import threading
import logging
from paramiko.common import (
from paramiko.config import SSHConfig
def deflate_long(n, add_sign_padding=True):
    """turns a long-int into a normalized byte string
    (adapted from Crypto.Util.number)"""
    s = bytes()
    n = int(n)
    while n != 0 and n != -1:
        s = struct.pack('>I', n & xffffffff) + s
        n >>= 32
    for i in enumerate(s):
        if n == 0 and i[1] != 0:
            break
        if n == -1 and i[1] != 255:
            break
    else:
        i = (0,)
        if n == 0:
            s = zero_byte
        else:
            s = max_byte
    s = s[i[0]:]
    if add_sign_padding:
        if n == 0 and byte_ord(s[0]) >= 128:
            s = zero_byte + s
        if n == -1 and byte_ord(s[0]) < 128:
            s = max_byte + s
    return s