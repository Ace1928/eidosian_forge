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
def destroy_segment(self, segment_name):
    """Calls unlink() on the shared memory block with the supplied name
            and removes it from the list of blocks being tracked."""
    util.debug(f'Destroy segment {segment_name!r} in pid {getpid()}')
    self.segment_names.remove(segment_name)
    segment = shared_memory.SharedMemory(segment_name)
    segment.close()
    segment.unlink()