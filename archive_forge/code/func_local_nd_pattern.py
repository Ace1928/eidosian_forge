import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def local_nd_pattern(*sizes, distance, p=2.0):
    d = local_nd_distance(*sizes, p=p)
    return d < distance