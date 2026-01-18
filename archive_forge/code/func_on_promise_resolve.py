from collections import namedtuple
from functools import partial
from threading import local
from .promise import Promise, async_instance, get_default_scheduler
def on_promise_resolve(v):
    async_instance.invoke(fn, scheduler)