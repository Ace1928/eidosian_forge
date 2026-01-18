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
def startRequestProcessing(self, interval=0.01):
    """Start listening for requests coming from the child process.
        This allows signals to be connected from the child process to the parent.
        """
    self.timer.timeout.connect(self.processRequests)
    self.timer.start(int(interval * 1000))