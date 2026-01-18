from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
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