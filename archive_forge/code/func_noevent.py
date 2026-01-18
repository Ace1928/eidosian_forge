import logging
import sys
import types
import threading
import inspect
from functools import wraps
from itertools import chain
from numba.core import config
def noevent(msg):
    pass