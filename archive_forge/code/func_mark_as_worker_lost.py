import copy
import errno
import itertools
import os
import platform
import signal
import sys
import threading
import time
import warnings
from collections import deque
from functools import partial
from . import cpu_count, get_context
from . import util
from .common import (
from .compat import get_errno, mem_rss, send_offset
from .einfo import ExceptionInfo
from .dummy import DummyProcess
from .exceptions import (
from time import monotonic
from queue import Queue, Empty
from .util import Finalize, debug, warning
def mark_as_worker_lost(self, job, exitcode):
    try:
        raise WorkerLostError('Worker exited prematurely: {0} Job: {1}.'.format(human_status(exitcode), job._job))
    except WorkerLostError:
        job._set(None, (False, ExceptionInfo()))
    else:
        pass