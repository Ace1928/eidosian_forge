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
def reporter(self, callback: ExtractCallback):
    while True:
        try:
            item: Optional[Tuple[str, str, str]] = self.q.get(timeout=1)
        except queue.Empty:
            pass
        else:
            if item is None:
                break
            elif item[0] == 's':
                callback.report_start(item[1], item[2])
            elif item[0] == 'u':
                callback.report_update(item[2])
            elif item[0] == 'e':
                callback.report_end(item[1], item[2])
            elif item[0] == 'pre':
                callback.report_start_preparation()
            elif item[0] == 'post':
                callback.report_postprocess()
            elif item[0] == 'w':
                callback.report_warning(item[1])
            else:
                pass
            self.q.task_done()