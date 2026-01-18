import sys
import threading
import signal
import array
import queue
import time
import types
import os
from os import getpid
from traceback import format_exc
from . import connection
from .context import reduction, get_spawning_popen, ProcessError
from . import pool
from . import process
from . import util
from . import get_context
def list_segments(self, c):
    """Returns a list of names of shared memory blocks that the Server
            is currently tracking."""
    return self.shared_memory_context.segment_names