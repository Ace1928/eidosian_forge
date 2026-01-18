import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def local_2d_distance(H, W, p=2.0):
    return local_nd_distance(H, W, p=p)