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
def terminate_broken(self, cause):
    executor = self.executor_reference()
    if executor is not None:
        executor._broken = 'A child process terminated abruptly, the process pool is not usable anymore'
        executor._shutdown_thread = True
        executor = None
    bpe = BrokenProcessPool('A process in the process pool was terminated abruptly while the future was running or pending.')
    if cause is not None:
        bpe.__cause__ = _RemoteTraceback(f"\n'''\n{''.join(cause)}'''")
    for work_id, work_item in self.pending_work_items.items():
        work_item.future.set_exception(bpe)
        del work_item
    self.pending_work_items.clear()
    for p in self.processes.values():
        p.terminate()
    self.call_queue._reader.close()
    if sys.platform == 'win32':
        self.call_queue._writer.close()
    self.join_executor_internals()