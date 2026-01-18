from functools import reduce
import torch
import operator
from torch.fx.tensor_type import Dyn, is_consistent, TensorType, is_more_precise
from typing import Callable, Dict
from torch.fx.node import Target, Node
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.fx.experimental.refinement_types import Equality
import itertools
from torch.fx.experimental.unification import Var  # type: ignore[attr-defined]
import sympy
def maxpool2d_check(typ, module_instance):
    """
    Applies the maxpool2d shape information to the input
    this affects the last two dimensions
    """
    new_type_list = list(typ.__args__)
    if len(new_type_list) == 4 or len(new_type_list) == 3:
        w_in = new_type_list[-1]
        h_in = new_type_list[-2]
        h_out = calculate_out_dimension(h_in, module_instance, 0)
        w_out = calculate_out_dimension(w_in, module_instance, 1)
        new_type_list[-1] = w_out
        new_type_list[-2] = h_out
        return TensorType(tuple(new_type_list))
    else:
        raise TypeError(f'Wrong size {typ} for {module_instance}')