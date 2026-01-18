import os
import signal
import sys
import pickle
from .exceptions import RestartFreqExceeded
from time import monotonic
from io import BytesIO
def maybe_setsignal(signum, handler):
    try:
        signal.signal(signum, handler)
    except (OSError, AttributeError, ValueError, RuntimeError):
        pass