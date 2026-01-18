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
def on_trace_ready(profiler: _PROFILER) -> None:
    if self.dirpath is not None:
        if self._export_to_chrome:
            handler = tensorboard_trace_handler(str(self.dirpath), self._prepare_filename(action_name=action_name, extension=''))
            handler(profiler)
        if self._export_to_flame_graph:
            path = os.path.join(self.dirpath, self._prepare_filename(action_name=action_name, extension='.stack'))
            assert isinstance(profiler, torch.autograd.profiler.profile)
            profiler.export_stacks(path, metric=self._metric)
    else:
        rank_zero_warn('The PyTorchProfiler failed to export trace as `dirpath` is None')