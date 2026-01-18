import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework
def register_signal_handler():
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except ValueError:
        pass