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
def send_to_client(self, address, result):
    with logerrors():
        if not isinstance(result, list):
            result = [result]
        with self._socket_lock:
            self.socket.send_multipart([address] + result)