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
def scheduler_hook(param):
    if param.grad is not None:
        scheduler_dict[param].step()