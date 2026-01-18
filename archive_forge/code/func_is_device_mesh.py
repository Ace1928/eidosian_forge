import inspect
from typing import Dict, List
import torch
from .. import variables
from ..exc import unimplemented
from ..utils import istype
from .base import VariableTracker
from .constant import ConstantVariable
@staticmethod
def is_device_mesh(value):
    if not DistributedVariable.is_available():
        return False
    from torch.distributed.device_mesh import DeviceMesh
    return istype(value, DeviceMesh)