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
@classmethod
def from_process(cls, process, name='stdio'):
    """Creates a new instance that receives messages from process.stdin, and sends
        them to process.stdout.
        """
    return cls(process.stdout, process.stdin, name)