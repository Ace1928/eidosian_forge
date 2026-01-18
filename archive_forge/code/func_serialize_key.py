import zmq
import logging
from itertools import chain
from bisect import bisect
import socket
from operator import add
from time import sleep, time
from toolz import accumulate, topk, pluck, merge, keymap
import uuid
from collections import defaultdict
from contextlib import contextmanager, suppress
from threading import Thread, Lock
from datetime import datetime
from multiprocessing import Process
import traceback
import sys
from .dict import Dict
from .file import File
from .buffer import Buffer
from . import core
from .core import Interface
from .file import File
def serialize_key(key):
    """

    >>> serialize_key('x')
    b'x'
    >>> serialize_key(('a', 'b', 1))
    b'a-|-b-|-1'
    """
    if isinstance(key, tuple):
        return tuple_sep.join(map(serialize_key, key))
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode()
    return str(key).encode()