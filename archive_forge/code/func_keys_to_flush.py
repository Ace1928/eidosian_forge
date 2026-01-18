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
def keys_to_flush(lengths, fraction=0.1, maxcount=100000):
    """ Which keys to remove

    >>> lengths = {'a': 20, 'b': 10, 'c': 15, 'd': 15,
    ...            'e': 10, 'f': 25, 'g': 5}
    >>> keys_to_flush(lengths, 0.5)
    ['f', 'a']
    """
    top = topk(max(len(lengths) // 2, 1), lengths.items(), key=1)
    total = sum(lengths.values())
    cutoff = min(maxcount, max(1, bisect(list(accumulate(add, pluck(1, top))), total * fraction)))
    result = [k for k, v in top[:cutoff]]
    assert result
    return result