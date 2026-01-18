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
@override
def finetune_function(self, pl_module: 'pl.LightningModule', epoch: int, optimizer: Optimizer) -> None:
    """Called when the epoch begins."""
    if epoch == self.unfreeze_backbone_at_epoch:
        current_lr = optimizer.param_groups[0]['lr']
        initial_backbone_lr = self.backbone_initial_lr if self.backbone_initial_lr is not None else current_lr * self.backbone_initial_ratio_lr
        self.previous_backbone_lr = initial_backbone_lr
        self.unfreeze_and_add_param_group(pl_module.backbone, optimizer, initial_backbone_lr, train_bn=self.train_bn, initial_denom_lr=self.initial_denom_lr)
        if self.verbose:
            log.info(f'Current lr: {round(current_lr, self.rounding)}, Backbone lr: {round(initial_backbone_lr, self.rounding)}')
    elif epoch > self.unfreeze_backbone_at_epoch:
        current_lr = optimizer.param_groups[0]['lr']
        next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
        next_current_backbone_lr = current_lr if self.should_align and next_current_backbone_lr > current_lr else next_current_backbone_lr
        optimizer.param_groups[-1]['lr'] = next_current_backbone_lr
        self.previous_backbone_lr = next_current_backbone_lr
        if self.verbose:
            log.info(f'Current lr: {round(current_lr, self.rounding)}, Backbone lr: {round(next_current_backbone_lr, self.rounding)}')