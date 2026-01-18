import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
def is_from_local(value):
    if not DistributedVariable.is_available():
        return False
    from torch.distributed._tensor import DTensor
    return inspect.isfunction(value) and value is DTensor.from_local