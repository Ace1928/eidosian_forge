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
def requested_stop(self):
    """If the work unit being ran has requested to be stopped."""
    return self._metrics['requested_stop']