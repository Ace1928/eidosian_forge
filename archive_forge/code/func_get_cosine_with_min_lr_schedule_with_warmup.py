import math
import warnings
from functools import partial
from typing import Callable, Iterable, Optional, Tuple, Union
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from .trainer_pt_utils import LayerWiseDummyOptimizer, LayerWiseDummyScheduler
from .trainer_utils import SchedulerType
from .utils import logging
from .utils.versions import require_version
def get_cosine_with_min_lr_schedule_with_warmup(optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float=0.5, last_epoch: int=-1, min_lr: float=None, min_lr_rate: float=None):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to min_lr, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr (`float`, *optional*):
            The minimum learning rate to reach after the cosine schedule.
        min_lr_rate (`float`, *optional*):
            The minimum learning rate as a ratio of the initial learning rate. If set, `min_lr` should not be set.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError('Only one of min_lr or min_lr_rate should be set')
    elif min_lr is not None:
        min_lr_rate = min_lr / optimizer.defaults['lr']
    elif min_lr_rate is None:
        raise ValueError('One of min_lr or min_lr_rate should be set through the `lr_scheduler_kwargs`')
    lr_lambda = partial(_get_cosine_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, num_cycles=num_cycles, min_lr_rate=min_lr_rate)
    return LambdaLR(optimizer, lr_lambda, last_epoch)