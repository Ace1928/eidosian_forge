import asyncio
import functools
import queue
import threading
import time
from typing import (
def gather_batch(request_queue: 'queue.Queue[Request]', batch_time: float, inter_event_time: float, max_batch_size: int, clock: Callable[[], float]=time.monotonic) -> Tuple[bool, Sequence[RequestPrepare]]:
    batch_start_time = clock()
    remaining_time = batch_time
    first_request = request_queue.get()
    if isinstance(first_request, RequestFinish):
        return (True, [])
    batch: List[RequestPrepare] = [first_request]
    while remaining_time > 0 and len(batch) < max_batch_size:
        try:
            request = request_queue.get(timeout=_clamp(x=inter_event_time, low=1e-12, high=remaining_time))
            if isinstance(request, RequestFinish):
                return (True, batch)
            batch.append(request)
            remaining_time = batch_time - (clock() - batch_start_time)
        except queue.Empty:
            break
    return (False, batch)