from multiprocessing import Process, Queue
from wandb_promise import Promise
from .utils import process
def queue_process(q):
    promise, fn, args, kwargs = q.get()
    process(promise, fn, args, kwargs)