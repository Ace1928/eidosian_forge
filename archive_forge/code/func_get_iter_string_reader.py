import asyncio
from collections import deque
import errno
import fcntl
import gc
import getpass
import glob as glob_module
import inspect
import logging
import os
import platform
import pty
import pwd
import re
import select
import signal
import stat
import struct
import sys
import termios
import textwrap
import threading
import time
import traceback
import tty
import warnings
import weakref
from asyncio import Queue as AQueue
from contextlib import contextmanager
from functools import partial
from importlib import metadata
from io import BytesIO, StringIO, UnsupportedOperation
from io import open as fdopen
from locale import getpreferredencoding
from queue import Empty, Queue
from shlex import quote as shlex_quote
from types import GeneratorType, ModuleType
from typing import Any, Dict, Type, Union
def get_iter_string_reader(stdin):
    """return an iterator that returns a chunk of a string every time it is
    called.  notice that even though bufsize_type might be line buffered, we're
    not doing any line buffering here.  that's because our StreamBufferer
    handles all buffering.  we just need to return a reasonable-sized chunk."""
    bufsize = 1024
    iter_str = (stdin[i:i + bufsize] for i in range(0, len(stdin), bufsize))
    return get_iter_chunk_reader(iter_str)