import math
import torch
from torch.utils import benchmark
from torch.utils.benchmark import FuzzedParameter, FuzzedTensor, ParameterAlias
def power_range(upper_bound, base):
    return (base ** i for i in range(int(math.log(upper_bound, base)) + 1))