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
def debugMsg(self, msg, *args):
    if hasattr(self, '_stdoutForwarder'):
        with self._stdoutForwarder.lock:
            with self._stderrForwarder.lock:
                RemoteEventHandler.debugMsg(self, msg, *args)
    else:
        RemoteEventHandler.debugMsg(self, msg, *args)