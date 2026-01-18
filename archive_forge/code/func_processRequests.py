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
def processRequests(self):
    try:
        Process.processRequests(self)
    except ClosedError:
        self.timer.stop()