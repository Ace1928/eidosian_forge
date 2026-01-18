from collections import deque
from threading import local
def drain_queue(self, queue):
    from .promise import Promise
    while queue:
        fn = queue.popleft()
        if isinstance(fn, Promise):
            fn._settle_promises()
            continue
        fn()