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
def flush_archive(self, fp, folder):
    compressor = folder.get_compressor()
    foutsize = compressor.flush(fp)
    if len(self.files) > 0:
        if 'maxsize' in self.header.files_info.files[self.last_file_index]:
            self.header.files_info.files[self.last_file_index]['maxsize'] += foutsize
        else:
            self.header.files_info.files[self.last_file_index]['maxsize'] = foutsize
    self.header.main_streams.packinfo.numstreams += 1
    if self.header.main_streams.packinfo.enable_digests:
        self.header.main_streams.packinfo.crcs.append(compressor.digest)
        self.header.main_streams.packinfo.digestdefined.append(True)
    self.header.main_streams.packinfo.packsizes.append(compressor.packsize)
    folder.unpacksizes = compressor.unpacksizes