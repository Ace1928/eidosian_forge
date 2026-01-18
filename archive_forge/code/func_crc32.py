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
def crc32(ch, crc):
    """Compute the CRC32 primitive on one byte."""
    return crc >> 8 ^ crctable[(crc ^ ch) & 255]