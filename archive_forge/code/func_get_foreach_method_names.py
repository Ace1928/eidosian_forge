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
def get_foreach_method_names(name):
    op_name = '_foreach_' + name
    inplace_op_name = op_name + '_'
    op = getattr(torch, op_name, None)
    inplace_op = getattr(torch, inplace_op_name, None)
    ref = getattr(torch, name, None)
    ref_inplace = getattr(torch.Tensor, name + '_', None)
    return (op, inplace_op, ref, ref_inplace)