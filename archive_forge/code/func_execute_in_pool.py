from multiprocessing.pool import ThreadPool
from threading import Thread
from wandb_promise import Promise
from .utils import process
def execute_in_pool(self, fn, *args, **kwargs):
    promise = Promise()
    self.pool.map(lambda input: process(*input), [(promise, fn, args, kwargs)])
    return promise