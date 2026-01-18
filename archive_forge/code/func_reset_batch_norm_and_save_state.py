from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union, cast
import torch
from torch import Tensor, nn
from torch.optim.swa_utils import SWALR
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import LRScheduler
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.strategies.fsdp import FSDPStrategy
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
def reset_batch_norm_and_save_state(self, pl_module: 'pl.LightningModule') -> None:
    """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154."""
    self.momenta = {}
    for module in pl_module.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        assert module.running_mean is not None
        module.running_mean = torch.zeros_like(module.running_mean, device=pl_module.device, dtype=module.running_mean.dtype)
        assert module.running_var is not None
        module.running_var = torch.ones_like(module.running_var, device=pl_module.device, dtype=module.running_var.dtype)
        self.momenta[module] = module.momentum
        module.momentum = None
        assert module.num_batches_tracked is not None
        module.num_batches_tracked *= 0