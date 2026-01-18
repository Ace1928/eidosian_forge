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
@staticmethod
def avg_fn(averaged_model_parameter: Tensor, model_parameter: Tensor, num_averaged: Tensor) -> Tensor:
    """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97."""
    return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)