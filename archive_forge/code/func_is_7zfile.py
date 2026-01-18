import collections.abc
import contextlib
import datetime
import errno
import functools
import io
import os
import pathlib
import queue
import re
import stat
import sys
import time
from multiprocessing import Process
from threading import Thread
from typing import IO, Any, BinaryIO, Collection, Dict, List, Optional, Tuple, Type, Union
import multivolumefile
from py7zr.archiveinfo import Folder, Header, SignatureHeader
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods, get_methods_names
from py7zr.exceptions import (
from py7zr.helpers import (
from py7zr.properties import DEFAULT_FILTERS, FILTER_DEFLATE64, MAGIC_7Z, get_default_blocksize, get_memory_limit
def is_7zfile(file: Union[BinaryIO, str, pathlib.Path]) -> bool:
    """Quickly see if a file is a 7Z file by checking the magic number.
    The file argument may be a filename or file-like object too.
    """
    result = False
    try:
        if (isinstance(file, BinaryIO) or isinstance(file, io.BufferedReader) or isinstance(file, io.IOBase)) and hasattr(file, 'read'):
            result = SevenZipFile._check_7zfile(file)
        elif isinstance(file, str):
            with open(file, 'rb') as fp:
                result = SevenZipFile._check_7zfile(fp)
        elif isinstance(file, pathlib.Path) or isinstance(file, pathlib.PosixPath) or isinstance(file, pathlib.WindowsPath):
            with file.open(mode='rb') as fp:
                result = SevenZipFile._check_7zfile(fp)
        else:
            raise TypeError('invalid type: file should be str, pathlib.Path or BinaryIO, but {}'.format(type(file)))
    except OSError:
        pass
    return result