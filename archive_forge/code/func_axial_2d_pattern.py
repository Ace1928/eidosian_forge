import math
from typing import List
import numpy as np
import torch
from xformers.components.attention.sparsity_config import (
def axial_2d_pattern(H, W):
    return axial_nd_pattern(H, W)