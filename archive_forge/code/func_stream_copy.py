from abc import abstractmethod
import contextlib
from functools import wraps
import getpass
import logging
import os
import os.path as osp
import pathlib
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
from urllib.parse import urlsplit, urlunsplit
import warnings
from typing import (
from .types import (
from gitdb.util import (  # noqa: F401  # @IgnorePep8
def stream_copy(source: BinaryIO, destination: BinaryIO, chunk_size: int=512 * 1024) -> int:
    """Copy all data from the source stream into the destination stream in chunks
    of size chunk_size.

    :return: Number of bytes written
    """
    br = 0
    while True:
        chunk = source.read(chunk_size)
        destination.write(chunk)
        br += len(chunk)
        if len(chunk) < chunk_size:
            break
    return br