import os
import time
from contextlib import contextmanager
from typing import Callable, Optional
def reset_time():
    global _start_time, _debug_indent
    _start_time = time.time()
    _debug_indent = 0