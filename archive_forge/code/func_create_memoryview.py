import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def create_memoryview(self):
    (arena, start, stop), size = self._state
    return memoryview(arena.buffer)[start:start + size]