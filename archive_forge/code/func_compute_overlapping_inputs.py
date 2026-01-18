import itertools
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
from torch import Tensor
from torch._logging import getArtifactLogger
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx.experimental.symbolic_shapes import is_concrete_int
from .functional_utils import _get_mutation_type
from .schemas import (
from .utils import strict_zip
def compute_overlapping_inputs(fwd_inputs, aliased_input_indices):
    actual_aliased_indices = set()
    for j in range(len(aliased_input_indices)):
        for i in range(j):
            i_ = aliased_input_indices[i]
            j_ = aliased_input_indices[j]
            if not _tensors_definitely_do_not_overlap(fwd_inputs[i_], fwd_inputs[j_]):
                actual_aliased_indices.add(i_)
                actual_aliased_indices.add(j_)
    return actual_aliased_indices