import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_picker(func, conv1dOpt, conv2dOpt, conv3dOpt):
    if func == F.conv1d:
        return conv1dOpt
    if func == F.conv2d:
        return conv2dOpt
    else:
        assert func == F.conv3d
        return conv3dOpt