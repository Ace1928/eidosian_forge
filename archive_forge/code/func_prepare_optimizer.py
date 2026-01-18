import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
def prepare_optimizer(self, optimizer: Optimizer) -> Optimizer:
    """Wraps optimizer to support automatic mixed precision.

        Args:
            optimizer (torch.optim.Optimizer): The DataLoader to prepare.

        Returns:
            A wrapped optimizer.
        """
    return _WrappedOptimizer(optimizer, scaler=self.scaler)