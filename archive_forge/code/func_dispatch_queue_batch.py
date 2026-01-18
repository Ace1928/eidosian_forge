from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def dispatch_queue_batch(loader, queue):
    keys = [l.key for l in queue]
    try:
        batch_promise = loader.batch_load_fn(keys)
    except Exception as e:
        failed_dispatch(loader, queue, e)
        return None
    if not batch_promise or not isinstance(batch_promise, Promise):
        failed_dispatch(loader, queue, TypeError('DataLoader must be constructed with a function which accepts Array<key> and returns Promise<Array<value>>, but the function did not return a Promise: {}.'.format(batch_promise)))
        return None

    def batch_promise_resolved(values):
        if not isinstance(values, Iterable):
            raise TypeError('DataLoader must be constructed with a function which accepts Array<key> and returns Promise<Array<value>>, but the function did not return a Promise of an Array: {}.'.format(values))
        if len(values) != len(keys):
            raise TypeError('DataLoader must be constructed with a function which accepts Array<key> and returns Promise<Array<value>>, but the function did not return a Promise of an Array of the same length as the Array of keys.\n\nKeys:\n{}\n\nValues:\n{}'.format(keys, values))
        for l, value in zip(queue, values):
            if isinstance(value, Exception):
                l.reject(value)
            else:
                l.resolve(value)
    batch_promise.then(batch_promise_resolved).catch(partial(failed_dispatch, loader, queue))