import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
@property
def self_privateuse1_time_total(self):
    if self.is_async or not self.use_device:
        return 0
    if self.device_type == DeviceType.CPU:
        return self.privateuse1_time_total - sum([child.privateuse1_time_total for child in self.cpu_children])
    else:
        assert self.device_type == DeviceType.CUDA
        return self.privateuse1_time_total