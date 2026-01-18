import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def read_stream(stream, logger=None, encoding=None):
    """
    Robustly reads a stream, sending a warning to a logger
    if some decoding error was raised.

    >>> read_stream(bytearray([65, 0xc7, 65, 10, 66]))  # doctest: +ELLIPSIS
    ['A...A', 'B']


    """
    default_encoding = encoding or locale.getdefaultlocale()[1] or 'UTF-8'
    logger = logger or fmlogger
    try:
        out = stream.decode(default_encoding)
    except UnicodeDecodeError as err:
        out = stream.decode(default_encoding, errors='replace')
        logger.warning('Error decoding string: %s', err)
    return out.splitlines()