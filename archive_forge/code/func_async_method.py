import time as _time
from sys import exit as sysexit
from os import _exit as osexit
from threading import Thread, Semaphore
from multiprocessing import Process, cpu_count
def async_method(*args, **kwargs):
    if config['POOLS'][config['POOL_NAME']]['threads'] == 0:
        return callee(*args, **kwargs)
    if not config['KILL_RECEIVED']:
        try:
            single = config['POOLS'][config['POOL_NAME']]['engine'](target=_run_via_pool, args=args, kwargs=kwargs, daemon=False)
        except Exception:
            single = config['POOLS'][config['POOL_NAME']]['engine'](target=_run_via_pool, args=args, kwargs=kwargs)
        config['TASKS'].append(single)
        single.start()
        return single