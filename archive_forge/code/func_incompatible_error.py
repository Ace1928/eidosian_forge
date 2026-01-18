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
def incompatible_error():
    raise ValueError(f'vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): out_dims is not compatible with the structure of `outputs`. out_dims has structure {tree_flatten(out_dims)[1]} but outputs has structure {output_spec}.')