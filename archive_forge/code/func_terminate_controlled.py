import os
import sys
import signal
import itertools
import logging
import threading
from _weakrefset import WeakSet
from multiprocessing import process as _mproc
def terminate_controlled(self):
    self._controlled_termination = True
    self.terminate()