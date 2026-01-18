import contextlib
from collections import namedtuple, defaultdict
from datetime import datetime
from dask.callbacks import Callback
def task_scheduled(self, key, now):
    self.scheduled[key] = now