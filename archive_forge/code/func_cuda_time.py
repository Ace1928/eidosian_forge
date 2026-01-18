import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
@property
def cuda_time(self):
    return 0.0 if self.count == 0 else 1.0 * self.cuda_time_total / self.count