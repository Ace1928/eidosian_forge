import numpy
from ray.util.collective.types import ReduceOp, torch_available
def get_nccl_runtime_version():
    return get_version()