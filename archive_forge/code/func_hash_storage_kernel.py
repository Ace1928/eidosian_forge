import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set
import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator
from torch.multiprocessing.reductions import StorageWeakRef
@lazy_compile(dynamic=True)
def hash_storage_kernel(x):
    a = torch.randint(-2 ** 31, 2 ** 31, x.shape, device=x.device, dtype=torch.int32).abs()
    a = (a % (2 ** 31 - 1) + 1).long()
    b = torch.randint(-2 ** 31, 2 ** 31, x.shape, device=x.device, dtype=torch.int32).abs().long()
    return prims.xor_sum((a * x + b).int(), [0])