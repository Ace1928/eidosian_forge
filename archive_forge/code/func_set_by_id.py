import collections
import dataclasses
import enum
import itertools as it
import logging
from typing import (
from typing_extensions import Literal
import torch
from torch._C import FunctionSchema
from torch._C._autograd import _ProfilerResult
from torch._C._profiler import (
from torch._utils import _element_size
from torch.profiler import _utils
def set_by_id(self, key: TensorKey, category: Category) -> None:
    self._values[key.id].by_id = category
    self._values[key.id]._by_id_keyset.add(key)