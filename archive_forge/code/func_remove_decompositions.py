import inspect
from collections import defaultdict
from functools import wraps
from itertools import chain
from typing import Callable, Dict, List, Sequence, Union
import torch
import torch.library
from torch._ops import HigherOrderOperator, OpOverload, OpOverloadPacket
from torch._prims_common import CustomOutParamAnnotation
from torch.utils import _pytree as pytree
import torch._decomp.decompositions
import torch._refs
def remove_decompositions(decompositions: Dict[torch._ops.OperatorBase, Callable], aten_ops: Sequence[Union[OpOverload, OpOverloadPacket]]) -> None:
    """
    Given a dictionary of decompositions obtained from get_decompositions(), removes
    operators associated with a list of operator overloads and overload packets passed
    as input. If the decomposition dictionary does not contain a decomposition that is
    specified to be removed, it is silently ignored.
    """
    for op in aten_ops:
        if isinstance(op, OpOverloadPacket):
            for overload_name in op.overloads():
                opo = getattr(op, overload_name)
                decompositions.pop(opo, None)
        elif isinstance(op, OpOverload):
            decompositions.pop(op, None)