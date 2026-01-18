import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
from torch.ao.quantization.backend_config import get_native_backend_config
import torch.ao.quantization.fx._lower_to_native_backend as \
import torch.ao.quantization.quantization_mappings as quantization_mappings
from .ns_types import NSNodeTargetType
from typing import Callable, Dict, List, Optional, Set, Tuple
def add_op_to_sets_of_related_ops(base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]], op: NSNodeTargetType, related_op: Optional[NSNodeTargetType]) -> None:
    if related_op is not None:
        for set_of_related_ops in base_name_to_sets_of_related_ops.values():
            if related_op in set_of_related_ops:
                set_of_related_ops.add(op)
                return
        raise AssertionError(f'{related_op} was not found')
    else:
        counter = 0
        while str(counter) in base_name_to_sets_of_related_ops:
            counter += 1
        base_name_to_sets_of_related_ops[str(counter)] = {op}