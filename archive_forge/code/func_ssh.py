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
@contrib('ssh')
def ssh(orig):
    """An ssh command for automatic password login"""

    class SessionContent(object):

        def __init__(self):
            self.chars = deque(maxlen=50000)
            self.lines = deque(maxlen=5000)
            self.line_chars = []
            self.last_line = ''
            self.cur_char = ''

        def append_char(self, char):
            if char == '\n':
                line = self.cur_line
                self.last_line = line
                self.lines.append(line)
                self.line_chars = []
            else:
                self.line_chars.append(char)
            self.chars.append(char)
            self.cur_char = char

        @property
        def cur_line(self):
            line = ''.join(self.line_chars)
            return line

    class SSHInteract(object):

        def __init__(self, prompt_match, pass_getter, out_handler, login_success):
            self.prompt_match = prompt_match
            self.pass_getter = pass_getter
            self.out_handler = out_handler
            self.login_success = login_success
            self.content = SessionContent()
            self.pw_entered = False
            self.success = False

        def __call__(self, char, stdin):
            self.content.append_char(char)
            if self.pw_entered and (not self.success):
                self.success = self.login_success(self.content)
            if self.success:
                return self.out_handler(self.content, stdin)
            if self.prompt_match(self.content):
                password = self.pass_getter()
                stdin.put(password + '\n')
                self.pw_entered = True

    def process(a, kwargs):
        real_out_handler = kwargs.pop('interact')
        password = kwargs.pop('password', None)
        login_success = kwargs.pop('login_success', None)
        prompt_match = kwargs.pop('prompt', None)
        prompt = 'Please enter SSH password: '
        if prompt_match is None:

            def prompt_match(content):
                return content.cur_line.endswith('password: ')
        if password is None:

            def pass_getter():
                return getpass.getpass(prompt=prompt)
        else:

            def pass_getter():
                return password.rstrip('\n')
        if login_success is None:

            def login_success(content):
                return True
        kwargs['_out'] = SSHInteract(prompt_match, pass_getter, real_out_handler, login_success)
        return (a, kwargs)
    cmd = orig.bake(_out_bufsize=0, _tty_in=True, _unify_ttys=True, _arg_preprocess=process)
    return cmd