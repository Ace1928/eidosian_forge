from collections import deque
from threading import local
def queue_tick(self, scheduler):
    if not self.is_tick_used:
        self.is_tick_used = True
        scheduler.call(self.drain_queues)