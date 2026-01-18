from collections import abc as container_abcs, defaultdict
from copy import deepcopy
from itertools import chain
import torch
import bitsandbytes.functional as F
def get_state_buffer(self, p, dtype=torch.float32):
    if not self.is_paged or p.numel() < 100000.0:
        return torch.zeros_like(p, dtype=dtype, device=p.device)
    else:
        buff = F.get_paged(*p.shape, dtype=dtype, device=p.device)
        F.fill(buff, 0)
        self.page_mng.paged_tensors.append(buff)
        return buff