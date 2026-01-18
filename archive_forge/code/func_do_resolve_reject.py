from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def do_resolve_reject(self, key, resolve, reject):
    self._queue.append(Loader(key=key, resolve=resolve, reject=reject))
    if len(self._queue) == 1:
        if self.batch:
            enqueue_post_promise_job(partial(dispatch_queue, self), self._scheduler)
        else:
            dispatch_queue(self)