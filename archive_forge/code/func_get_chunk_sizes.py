import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def get_chunk_sizes(total_elems, chunk_size):
    n_chunks = n_chunks = total_elems // chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    remainder = total_elems % chunk_size
    if remainder != 0:
        chunk_sizes.append(remainder)
    return chunk_sizes