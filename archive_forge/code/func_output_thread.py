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
def output_thread(log, stdout, stderr, timeout_event, is_alive, quit_thread, stop_output_event, output_complete):
    """this function is run in a separate thread.  it reads from the
    process's stdout stream (a streamreader), and waits for it to claim that
    its done"""
    poller = Poller()
    if stdout is not None:
        poller.register_read(stdout)
    if stderr is not None:
        poller.register_read(stderr)
    while poller:
        changed = no_interrupt(poller.poll, 0.1)
        for f, events in changed:
            if events & (POLLER_EVENT_READ | POLLER_EVENT_HUP):
                log.debug('%r ready to be read from', f)
                done = f.read()
                if done:
                    poller.unregister(f)
            elif events & POLLER_EVENT_ERROR:
                pass
        if timeout_event and timeout_event.is_set():
            break
        if stop_output_event.is_set():
            break
    alive, _ = is_alive()
    while alive:
        quit_thread.wait(1)
        alive, _ = is_alive()
    if stdout:
        stdout.close()
    if stderr:
        stderr.close()
    output_complete()