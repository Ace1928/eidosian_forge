import os
import sys
import signal
import itertools
import threading
from _weakrefset import WeakSet
@property
def ident(self):
    return self._pid