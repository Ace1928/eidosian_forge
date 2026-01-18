import functools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpStrategy
from torch.distributed._tensor.placement_types import (
def register_prop_rule(op, schema_info=None):

    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._op_dispatcher.sharding_propagator.register_sharding_prop_rule(overload, impl, schema_info)
        return impl
    return wrapper