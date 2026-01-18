import codecs
import errno
import os
import re
import stat
import sys
import time
from functools import partial
from typing import Dict, List
from .lazy_import import lazy_import
import locale
import ntpath
import posixpath
import select
import shutil
from shutil import rmtree
import socket
import subprocess
import unicodedata
from breezy import (
from breezy.i18n import gettext
from hashlib import md5
from hashlib import sha1 as sha
import breezy
from . import errors
def pumpfile(from_file, to_file, read_length=-1, buff_size=32768, report_activity=None, direction='read'):
    """Copy contents of one file to another.

    The read_length can either be -1 to read to end-of-file (EOF) or
    it can specify the maximum number of bytes to read.

    The buff_size represents the maximum size for each read operation
    performed on from_file.

    :param report_activity: Call this as bytes are read, see
        Transport._report_activity
    :param direction: Will be passed to report_activity

    :return: The number of bytes copied.
    """
    length = 0
    if read_length >= 0:
        while read_length > 0:
            num_bytes_to_read = min(read_length, buff_size)
            block = from_file.read(num_bytes_to_read)
            if not block:
                break
            if report_activity is not None:
                report_activity(len(block), direction)
            to_file.write(block)
            actual_bytes_read = len(block)
            read_length -= actual_bytes_read
            length += actual_bytes_read
    else:
        while True:
            block = from_file.read(buff_size)
            if not block:
                break
            if report_activity is not None:
                report_activity(len(block), direction)
            to_file.write(block)
            length += len(block)
    return length