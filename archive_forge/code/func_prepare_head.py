import math
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once
from parlai.utils.torch import neginf, PipelineHelper
def prepare_head(tensor):
    bsz, seq_len, _ = tensor.size()
    tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
    tensor = tensor.transpose(1, 2).contiguous().view(batch_size * n_heads, seq_len, dim_per_head)
    return tensor