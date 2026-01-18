import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
def flag_executor_shutting_down(self):
    executor = self.executor_reference()
    if executor is not None:
        executor._shutdown_thread = True
        if executor._cancel_pending_futures:
            new_pending_work_items = {}
            for work_id, work_item in self.pending_work_items.items():
                if not work_item.future.cancel():
                    new_pending_work_items[work_id] = work_item
            self.pending_work_items = new_pending_work_items
            while True:
                try:
                    self.work_ids_queue.get_nowait()
                except queue.Empty:
                    break
            executor._cancel_pending_futures = False