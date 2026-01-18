import collections
import collections.abc
import math
import operator
import unittest
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from itertools import product
from typing import Any, Callable, Iterable, List, Optional, Tuple
from torchgen.utils import dataclass_repr
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import (
from torch.testing._internal.opinfo import utils
def generate_elementwise_unary_extremal_value_tensors(op, *, device, dtype, requires_grad=False):
    for sample in generate_elementwise_binary_extremal_value_tensors(op, device=device, dtype=dtype, requires_grad=requires_grad):
        yield SampleInput(sample.input, kwargs=op.sample_kwargs(device, dtype, sample.input)[0])