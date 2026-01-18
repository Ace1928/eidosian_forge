import math
import torch
import torch.nn as nn
import torch.nn.functional as F
@torch.jit.script
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x)).to(dtype=x.dtype)