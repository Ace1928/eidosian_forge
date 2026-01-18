import logging
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union
import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
@staticmethod
def filter_on_optimizer(optimizer: Optimizer, params: Iterable) -> List:
    """This function is used to exclude any parameter which already exists in this optimizer.

        Args:
            optimizer: Optimizer used for parameter exclusion
            params: Iterable of parameters used to check against the provided optimizer

        Returns:
            List of parameters not contained in this optimizer param groups

        """
    out_params = []
    removed_params = []
    for param in params:
        if not any((torch.equal(p, param) for group in optimizer.param_groups for p in group['params'])):
            out_params.append(param)
        else:
            removed_params.append(param)
    if removed_params:
        rank_zero_warn(f'The provided params to be frozen already exist within another group of this optimizer. Those parameters will be skipped.\nHINT: Did you init your optimizer in `configure_optimizer` as such:\n {type(optimizer)}(filter(lambda p: p.requires_grad, self.parameters()), ...) ')
    return out_params