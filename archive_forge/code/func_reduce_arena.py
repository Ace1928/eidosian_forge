import bisect
from collections import defaultdict
import mmap
import os
import sys
import tempfile
import threading
from .context import reduction, assert_spawning
from . import util
def reduce_arena(a):
    if a.fd == -1:
        raise ValueError('Arena is unpicklable because forking was enabled when it was created')
    return (rebuild_arena, (a.size, reduction.DupFd(a.fd)))