from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
def log_message_and_reraise_exception(format_string='', *args, **kwargs):
    if format_string:
        format_string += '\n\n'
    format_string += '{name} -->\n{raw_lines}'
    raw_lines = b''.join(raw_chunks).split(b'\n')
    raw_lines = '\n'.join((repr(line) for line in raw_lines))
    log.reraise_exception(format_string, *args, name=self.name, raw_lines=raw_lines, **kwargs)