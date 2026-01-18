import atexit
import inspect
import multiprocessing.connection
import os
import signal
import subprocess
import sys
import time
import pickle
from ..Qt import QT_LIB, mkQApp
from ..util import cprint  # color printing for debugging
from .remoteproxy import (
import threading
def eventLoop(self):
    while True:
        try:
            self.processRequests()
            time.sleep(0.01)
        except ClosedError:
            break
        except:
            print('Error occurred in forked event loop:')
            sys.excepthook(*sys.exc_info())
    sys.exit(0)