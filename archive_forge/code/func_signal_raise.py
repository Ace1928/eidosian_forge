from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
def signal_raise(signum: int) -> None:
    signal.pthread_kill(threading.get_ident(), signum)