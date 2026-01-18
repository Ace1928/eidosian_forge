import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
@property
def self_cpu_memory_usage(self):
    if self.is_async or self.device_type != DeviceType.CPU:
        return 0
    return self.cpu_memory_usage - sum([child.cpu_memory_usage for child in self.cpu_children])