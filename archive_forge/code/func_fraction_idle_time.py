import functools
import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, List
from torch.autograd import _KinetoEvent
from torch.autograd.profiler import profile
from torch.profiler import DeviceType
@property
def fraction_idle_time(self):
    if self.duration_time_ns == 0:
        return 0.0
    return self.idle_time_ns / self.duration_time_ns