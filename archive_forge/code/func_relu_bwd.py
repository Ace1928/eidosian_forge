import math
import torch
import torch.nn as nn
import torch.nn.functional as F
@torch.jit.script
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)