from collections import deque
from threading import local
def drain_queue_until_resolved(self, promise):
    from .promise import Promise
    queue = self.normal_queue
    while queue:
        if not promise.is_pending:
            return
        fn = queue.popleft()
        if isinstance(fn, Promise):
            fn._settle_promises()
            continue
        fn()
    self.reset()
    self.have_drained_queues = True
    self.drain_queue(self.late_queue)