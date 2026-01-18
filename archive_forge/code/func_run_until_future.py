import queue
import sys
import threading
from concurrent.futures import Executor, Future
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union
def run_until_future(self, future: 'Future[Any]') -> None:
    """
        Runs the code in the work queue until a result is available from the future.
        Should be run from the thread the executor is initialised in.
        """
    if threading.current_thread() != self._work_thread:
        raise RuntimeError('You cannot run CurrentThreadExecutor from a different thread')
    future.add_done_callback(self._work_queue.put)
    try:
        while True:
            work_item = self._work_queue.get()
            if work_item is future:
                return
            assert isinstance(work_item, _WorkItem)
            work_item.run()
            del work_item
    finally:
        self._broken = True