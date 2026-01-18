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
def successes(self):
    """How many times the periodic callback ran successfully."""
    return self._metrics['successes']