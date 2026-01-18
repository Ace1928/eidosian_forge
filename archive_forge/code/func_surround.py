from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.fp16 import fp16_optimizer_wrapper
from parlai.utils.torch import neginf
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import clip_grad_norm_
def surround(idx_vector, start_idx, end_idx):
    """
    Surround the vector by start_idx and end_idx.
    """
    start_tensor = idx_vector.new_tensor([start_idx])
    end_tensor = idx_vector.new_tensor([end_idx])
    return torch.cat([start_tensor, idx_vector, end_tensor], 0)