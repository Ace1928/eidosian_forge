import collections
from typing import Deque, Optional
import torch
def dequeue_if_needed(self) -> Optional[torch.cuda.Event]:
    """Dequeues a single event if the limit is reached."""
    if len(self._queue) >= self._max_num_inflight_all_gathers:
        return self._dequeue()
    return None