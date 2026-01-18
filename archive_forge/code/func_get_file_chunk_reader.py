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
def get_file_chunk_reader(stdin):
    bufsize = 1024

    def fn():
        is_real_file = True
        try:
            stdin.fileno()
        except UnsupportedOperation:
            is_real_file = False
        if is_real_file and hasattr(stdin, 'fileno'):
            poller = Poller()
            poller.register_read(stdin)
            changed = poller.poll(0.1)
            ready = False
            for fd, events in changed:
                if events & (POLLER_EVENT_READ | POLLER_EVENT_HUP):
                    ready = True
            if not ready:
                raise NotYetReadyToRead
        chunk = stdin.read(bufsize)
        if not chunk:
            raise DoneReadingForever
        else:
            return chunk
    return fn