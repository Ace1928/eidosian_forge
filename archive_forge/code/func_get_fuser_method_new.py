import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from typing import Any, Union, Callable, List, Tuple, Dict, Optional, Type
from torch.ao.quantization.utils import Pattern, get_combined_dict, MatchAllNode
import itertools
def get_fuser_method_new(op_pattern: Pattern, fuser_method_mapping: Dict[Pattern, Union[nn.Sequential, Callable]]):
    """Get fuser method.

    This will be made default after we deprecate the get_fuser_method
    Would like to implement this first and have a separate PR for deprecation
    """
    op_patterns = _get_valid_patterns(op_pattern)
    fuser_method = None
    for op_pattern in op_patterns:
        fuser_method = fuser_method_mapping.get(op_pattern, None)
        if fuser_method is not None:
            break
    assert fuser_method is not None, f'did not find fuser method for: {op_pattern} '
    return fuser_method