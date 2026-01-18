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
def handle_command_exit_code(self, code):
    """here we determine if we had an exception, or an error code that we
        weren't expecting to see.  if we did, we create and raise an exception
        """
    ca = self.call_args
    exc_class = get_exc_exit_code_would_raise(code, ca['ok_code'], ca['piped'])
    if exc_class:
        exc = exc_class(self.ran, self.process.stdout, self.process.stderr, ca['truncate_exc'])
        raise exc