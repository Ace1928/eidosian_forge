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
def fetch_next_run(self, index):
    for next_run, a_index in self._ordering:
        if a_index == index:
            return next_run
    return None