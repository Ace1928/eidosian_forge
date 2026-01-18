import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def local_nd_gaussian_distribution(*sizes, sigma=1):
    d = local_nd_distance(*sizes, p=2.0) ** 2
    d = torch.exp(-0.5 * sigma ** (-2.0) * d)
    return d