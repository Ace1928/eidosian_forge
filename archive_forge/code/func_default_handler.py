import signal
import weakref
from functools import wraps
def default_handler(unused_signum, unused_frame):
    pass