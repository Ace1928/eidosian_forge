import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
@property
def elapsed_waiting(self):
    """Total amount of time the periodic callback has waited to run for."""
    return self._metrics['elapsed_waiting']