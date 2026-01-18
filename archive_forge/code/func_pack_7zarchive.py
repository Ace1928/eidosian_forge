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
def pack_7zarchive(base_name, base_dir, owner=None, group=None, dry_run=None, logger=None):
    """
    Function for registering with shutil.register_archive_format().
    """
    target_name = '{}.7z'.format(base_name)
    with SevenZipFile(target_name, mode='w') as archive:
        archive.writeall(path=base_dir)
    return target_name