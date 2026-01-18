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
def archiveinfo(self) -> ArchiveInfo:
    total_uncompressed = functools.reduce(lambda x, y: x + y, [f.uncompressed for f in self.files])
    if isinstance(self.fp, multivolumefile.MultiVolume):
        fname = self.fp.name
        fstat = self.fp.stat()
    else:
        fname = self.filename
        assert fname is not None
        fstat = os.stat(fname)
    return ArchiveInfo(fname, fstat, self.header.size, self._get_method_names(), self._is_solid(), len(self.header.main_streams.unpackinfo.folders), total_uncompressed)