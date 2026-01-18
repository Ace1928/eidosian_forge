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
def get_rc_exc(rc):
    """takes a exit code or negative signal number and produces an exception
    that corresponds to that return code.  positive return codes yield
    ErrorReturnCode exception, negative return codes yield SignalException

    we also cache the generated exception so that only one signal of that type
    exists, preserving identity"""
    try:
        return rc_exc_cache[rc]
    except KeyError:
        pass
    if rc >= 0:
        name = f'ErrorReturnCode_{rc}'
        base = ErrorReturnCode
    else:
        name = f'SignalException_{SIGNAL_MAPPING[abs(rc)]}'
        base = SignalException
    exc = ErrorReturnCodeMeta(name, (base,), {'exit_code': rc})
    rc_exc_cache[rc] = exc
    return exc