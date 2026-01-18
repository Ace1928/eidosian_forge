from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def dispatch_queue(loader):
    """
    Given the current state of a Loader instance, perform a batch load
    from its current queue.
    """
    queue = loader._queue
    loader._queue = []
    max_batch_size = loader.max_batch_size
    if max_batch_size and max_batch_size < len(queue):
        chunks = get_chunks(queue, max_batch_size)
        for chunk in chunks:
            dispatch_queue_batch(loader, chunk)
    else:
        dispatch_queue_batch(loader, queue)