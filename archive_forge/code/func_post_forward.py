from typing import List
import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm
def post_forward(module: _BatchNorm, input: Tensor, result: Tensor) -> None:
    if torch.is_grad_enabled():
        return
    module.track_running_stats = module._track_running_stats_backup