import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
@property
def cuda_time_total(self):
    if self.is_async or self.use_device:
        return 0
    if self.device_type == DeviceType.CPU:
        if not self.is_legacy:
            return sum((kinfo.duration for kinfo in self.kernels)) + sum((ch.cuda_time_total for ch in self.cpu_children))
        else:
            return sum((kinfo.duration for kinfo in self.kernels))
    else:
        assert self.device_type == DeviceType.CUDA
        return self.time_range.elapsed_us()