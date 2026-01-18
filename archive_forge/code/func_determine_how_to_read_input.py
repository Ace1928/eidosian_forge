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
def determine_how_to_read_input(input_obj):
    """given some kind of input object, return a function that knows how to
    read chunks of that input object.

    each reader function should return a chunk and raise a DoneReadingForever
    exception, or return None, when there's no more data to read

    NOTE: the function returned does not need to care much about the requested
    buffering type (eg, unbuffered vs newline-buffered).  the StreamBufferer
    will take care of that.  these functions just need to return a
    reasonably-sized chunk of data."""
    if isinstance(input_obj, Queue):
        log_msg = 'queue'
        get_chunk = get_queue_chunk_reader(input_obj)
    elif callable(input_obj):
        log_msg = 'callable'
        get_chunk = get_callable_chunk_reader(input_obj)
    elif hasattr(input_obj, 'read'):
        log_msg = 'file descriptor'
        get_chunk = get_file_chunk_reader(input_obj)
    elif isinstance(input_obj, str):
        log_msg = 'string'
        get_chunk = get_iter_string_reader(input_obj)
    elif isinstance(input_obj, bytes):
        log_msg = 'bytes'
        get_chunk = get_iter_string_reader(input_obj)
    elif isinstance(input_obj, GeneratorType):
        log_msg = 'generator'
        get_chunk = get_iter_chunk_reader(iter(input_obj))
    elif input_obj is None:
        log_msg = 'None'

        def raise_():
            raise DoneReadingForever
        get_chunk = raise_
    else:
        try:
            it = iter(input_obj)
        except TypeError:
            raise Exception('unknown input object')
        else:
            log_msg = 'general iterable'
            get_chunk = get_iter_chunk_reader(it)
    return (get_chunk, log_msg)