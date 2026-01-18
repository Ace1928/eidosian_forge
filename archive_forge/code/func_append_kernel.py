import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def append_kernel(self, name, device, duration):
    assert self.device_type == DeviceType.CPU
    self.kernels.append(Kernel(name, device, duration))