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
def output_iterator_validator(passed_kwargs, merged_kwargs):
    invalid = []
    if passed_kwargs.get('no_out') and passed_kwargs.get('iter') in (True, 'out'):
        error = 'You cannot iterate over output if there is no output'
        invalid.append((('no_out', 'iter'), error))
    return invalid