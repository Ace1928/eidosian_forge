from __future__ import print_function, division, absolute_import
import asyncio
import concurrent.futures
import contextlib
import time
from uuid import uuid4
import weakref
from .parallel import parallel_config
from .parallel import AutoBatchingMixin, ParallelBackendBase
def start_call(self):
    self._continue = True
    self.client.loop.add_callback(self._collect)
    self.call_data_futures = _WeakKeyDictionary()