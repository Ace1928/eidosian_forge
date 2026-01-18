import binascii
import importlib.util
import io
import itertools
import os
import posixpath
import shutil
import stat
import struct
import sys
import threading
import time
import contextlib
import pathlib
def decrypter(data):
    """Decrypt a bytes object."""
    result = bytearray()
    append = result.append
    for c in data:
        k = key2 | 2
        c ^= k * (k ^ 1) >> 8 & 255
        update_keys(c)
        append(c)
    return bytes(result)