import bisect
import itertools
import math
from collections import defaultdict, namedtuple
from operator import attrgetter
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.autograd import DeviceType
def auto_scale_flops(flops):
    flop_headers = ['FLOPs', 'KFLOPs', 'MFLOPs', 'GFLOPs', 'TFLOPs', 'PFLOPs']
    assert flops > 0
    log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
    assert log_flops >= 0 and log_flops < len(flop_headers)
    return (pow(10, math.floor(log_flops) * -3.0), flop_headers[int(log_flops)])