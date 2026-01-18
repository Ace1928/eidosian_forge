import bisect
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import islice
from operator import itemgetter
from time import time
from typing import Mapping, Optional  # noqa
from weakref import WeakSet, ref
from kombu.clocks import timetuple
from kombu.utils.objects import cached_property
from celery import states
from celery.utils.functional import LRUCache, memoize, pass1
from celery.utils.log import get_logger
def rebuild_taskheap(self, timetuple=timetuple):
    heap = self._taskheap[:] = [timetuple(t.clock, t.timestamp, t.origin, ref(t)) for t in self.tasks.values()]
    heap.sort()