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
@functools.wraps(fn)
def compile_hook(*args, **kwargs):
    compiled_fn = torch.compile(fn, **compile_kwargs)
    globals()[fn.__name__] = functools.wraps(fn)(compiled_fn)
    return compiled_fn(*args, **kwargs)