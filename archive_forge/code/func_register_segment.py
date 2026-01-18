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
def register_segment(self, segment_name):
    """Adds the supplied shared memory block name to tracker."""
    util.debug(f'Register segment {segment_name!r} in pid {getpid()}')
    self.segment_names.append(segment_name)