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
def get_file_chunk_consumer(handler, decode_errors):
    if getattr(handler, 'encoding', None):

        def encode(chunk):
            return chunk.decode(handler.encoding, decode_errors)
    else:

        def encode(chunk):
            return chunk
    if hasattr(handler, 'flush'):
        flush = handler.flush
    else:

        def flush():
            return None

    def process(chunk):
        handler.write(encode(chunk))
        flush()
        return False

    def finish():
        flush()
    return (process, finish)