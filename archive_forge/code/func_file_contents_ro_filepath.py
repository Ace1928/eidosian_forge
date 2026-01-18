import binascii
import os
import mmap
import sys
import time
import errno
from io import BytesIO
from smmap import (
import hashlib
from gitdb.const import (
def file_contents_ro_filepath(filepath, stream=False, allow_mmap=True, flags=0):
    """Get the file contents at filepath as fast as possible

    :return: random access compatible memory of the given filepath
    :param stream: see ``file_contents_ro``
    :param allow_mmap: see ``file_contents_ro``
    :param flags: additional flags to pass to os.open
    :raise OSError: If the file could not be opened

    **Note** for now we don't try to use O_NOATIME directly as the right value needs to be
    shared per database in fact. It only makes a real difference for loose object
    databases anyway, and they use it with the help of the ``flags`` parameter"""
    fd = os.open(filepath, os.O_RDONLY | getattr(os, 'O_BINARY', 0) | flags)
    try:
        return file_contents_ro(fd, stream, allow_mmap)
    finally:
        close(fd)