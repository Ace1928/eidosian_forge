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
def get_unmatchable_types_map() -> Dict[str, Set[NSNodeTargetType]]:
    FUNS_UNMATCHABLE: Set[NSNodeTargetType] = {torch.quantize_per_tensor, operator.getitem}
    MODS_UNMATCHABLE: Set[NSNodeTargetType] = {nn.Identity}
    METHS_UNMATCHABLE: Set[NSNodeTargetType] = {'to', 'dequantize', 'reshape', 'view', 'unsqueeze_', 'unsqueeze', 'transpose', 'squeeze_', 'squeeze', 'size', 'shape', 'resize_', 'repeat_interleave', 'repeat', 'permute', 'numel', 'mean', 'detach_', 'detach', 'contiguous', 'clamp', 'chunk'}
    return {'funs_unmatchable': FUNS_UNMATCHABLE, 'mods_unmatchable': MODS_UNMATCHABLE, 'meths_unmatchable': METHS_UNMATCHABLE}