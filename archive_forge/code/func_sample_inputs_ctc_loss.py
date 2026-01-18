from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_device_type import \
from torch.testing._internal.common_cuda import (
from torch.testing._internal.common_utils import (
import torch._refs as refs  # noqa: F401
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims  # noqa: F401
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import (  # noqa: F401
from torch.testing._internal.opinfo.refs import (  # NOQA: F401
from torch.testing._internal.opinfo.utils import (
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import (
from torch.testing._internal.opinfo.definitions.special import (
from torch.testing._internal.opinfo.definitions._masked import (
from torch.testing._internal.opinfo.definitions.sparse import (
def sample_inputs_ctc_loss(op_info, device, dtype, requires_grad, **kwargs):
    input_length = 50
    batch = 16
    num_char = 20
    target_length = 30

    def make_log_probs(s):
        t = make_tensor(s, device=device, dtype=dtype)
        log_probs = t.log_softmax(2).to(device=device, dtype=dtype).detach().requires_grad_(requires_grad=requires_grad)
        return log_probs
    reductions = ('none', 'mean', 'sum')
    zero_inf = (True, False)
    lengths_type = (list, torch.Tensor)
    for r, z, lt in product(reductions, zero_inf, lengths_type):
        log_probs = make_log_probs((input_length, batch, num_char))
        targets = torch.randint(1, num_char, (batch, target_length), dtype=torch.long, device=device)
        input_lengths = torch.full((batch,), input_length, dtype=torch.long, device=device)
        target_lengths = torch.randint(10, target_length, (batch,), dtype=torch.long, device=device)
        if lt is list and r in ['none', 'sum']:
            input_lengths = input_lengths.tolist()
            target_lengths = target_lengths.tolist()
        yield SampleInput(log_probs, args=(targets, input_lengths, target_lengths), kwargs=dict(reduction=r, zero_infinity=z))