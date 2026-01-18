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
def determine_how_to_feed_output(handler, encoding, decode_errors):
    if callable(handler):
        process, finish = get_callback_chunk_consumer(handler, encoding, decode_errors)
    elif isinstance(handler, BytesIO):
        process, finish = get_cstringio_chunk_consumer(handler)
    elif isinstance(handler, StringIO):
        process, finish = get_stringio_chunk_consumer(handler, encoding, decode_errors)
    elif hasattr(handler, 'write'):
        process, finish = get_file_chunk_consumer(handler, decode_errors)
    else:
        try:
            handler = int(handler)
        except (ValueError, TypeError):

            def process(chunk):
                return False

            def finish():
                return None
        else:
            process, finish = get_fd_chunk_consumer(handler, decode_errors)
    return (process, finish)