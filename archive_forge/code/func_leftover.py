import logging
import time
from monotonic import monotonic as now  # noqa
def leftover(self):
    if self.duration is None:
        return None
    return max(0.0, self.duration - self.elapsed())