import collections
import itertools
import os
import queue
import threading
import time
import traceback
import types
import warnings
from . import util
from . import get_context, TimeoutError
from .connection import wait
def starmapstar(args):
    return list(itertools.starmap(args[0], args[1]))