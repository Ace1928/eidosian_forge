import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version
def get_lr(self):
    opt = self.optimizer
    lrs = [opt._get_lr(group, opt.state[group['params'][0]]) for group in opt.param_groups if group['params'][0].grad is not None]
    if len(lrs) == 0:
        lrs = self.base_lrs
    return lrs