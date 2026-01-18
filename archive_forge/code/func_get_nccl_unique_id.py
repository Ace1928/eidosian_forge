import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_nccl_unique_id():
    return nccl.get_unique_id()