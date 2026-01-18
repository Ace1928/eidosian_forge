import os
import gc
import sys
import queue
import struct
import weakref
import warnings
import itertools
import traceback
import threading
from time import time, sleep
import multiprocessing as mp
from functools import partial
from pickle import PicklingError
from concurrent.futures import Executor
from concurrent.futures._base import LOGGER
from concurrent.futures.process import BrokenProcessPool as _BPPException
from multiprocessing.connection import wait
from ._base import Future
from .backend import get_context
from .backend.context import cpu_count, _MAX_WINDOWS_WORKERS
from .backend.queues import Queue, SimpleQueue
from .backend.reduction import set_loky_pickler, get_loky_pickler_name
from .backend.utils import kill_process_tree, get_exitcodes_terminated_worker
from .initializers import _prepare_initializer
def flag_as_broken(self, broken):
    with self.shutdown_lock:
        self.shutdown = True
        self.broken = broken