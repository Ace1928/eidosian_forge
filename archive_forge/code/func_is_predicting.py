import inspect
import logging
import os
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Dict, List, Optional, Type, Union
import torch
from torch import Tensor, nn
from torch.autograd.profiler import EventList, record_function
from torch.profiler import ProfilerAction, ProfilerActivity, tensorboard_trace_handler
from torch.utils.hooks import RemovableHandle
from typing_extensions import override
from lightning_fabric.accelerators.cuda import is_cuda_available
from pytorch_lightning.profilers.profiler import Profiler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_warn
@property
def is_predicting(self) -> bool:
    assert self._current_action is not None
    return self._current_action.endswith('predict_step')