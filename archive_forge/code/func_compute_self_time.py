import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
def compute_self_time(self):
    """
        Computes event's self time(total time - time in child ops).
        """
    assert self.profile.kineto_results is not None
    stack = deque(self.profile.kineto_results.experimental_event_tree())
    while stack:
        curr_event = stack.pop()
        self_time = curr_event.duration_time_ns
        for child_event in curr_event.children:
            self_time -= child_event.duration_time_ns
            stack.append(child_event)
        assert EventKey(curr_event) not in self.metrics, f'Duplicate id: {curr_event.id}, {curr_event.name}'
        self.metrics[EventKey(curr_event)] = EventMetrics(self_time_ns=self_time)
        self.metrics[EventKey(curr_event)].duration_time_ns = curr_event.duration_time_ns