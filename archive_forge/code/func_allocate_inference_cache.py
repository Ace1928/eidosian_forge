from functools import partial
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
    return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)