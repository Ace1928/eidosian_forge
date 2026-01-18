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
def special_kwarg_validator(passed_kwargs, merged_kwargs, invalid_list):
    s1 = set(passed_kwargs.keys())
    invalid_args = []
    for elem in invalid_list:
        if callable(elem):
            fn = elem
            ret = fn(passed_kwargs, merged_kwargs)
            invalid_args.extend(ret)
        else:
            elem, error_msg = elem
            if s1.issuperset(elem):
                invalid_args.append((elem, error_msg))
    return invalid_args