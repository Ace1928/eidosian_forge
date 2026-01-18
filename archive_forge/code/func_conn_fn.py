import sys
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
def conn_fn():
    priority = None
    conn = None
    try:
        priority, conn = self._connections.get()
        if conn is None:
            conn = self._create_connection()
        conn_args = (conn,) + args
        return fn(*conn_args, **kwargs)
    finally:
        if priority is not None:
            self._connections.put((priority, conn))