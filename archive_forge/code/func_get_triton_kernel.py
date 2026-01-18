import dataclasses
import tempfile
from collections import defaultdict
import torch
from torch.autograd import DeviceType
from .utils import create_bandwidth_info_str, do_bench, get_num_bytes
def get_triton_kernel(mod):
    from torch._inductor.triton_heuristics import CachingAutotuner
    cand_list = [v for k, v in mod.__dict__.items() if k.startswith('triton_') and isinstance(v, CachingAutotuner)]
    assert len(cand_list) == 1
    return cand_list[0]