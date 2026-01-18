import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
def calltree(self, file):
    """Output profiling data in calltree format (for KCacheGrind)."""
    _CallTreeFilter(self.data).output(file)