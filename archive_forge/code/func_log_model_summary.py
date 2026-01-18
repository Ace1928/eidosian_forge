import contextlib
import logging
import os
from argparse import Namespace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_only
@rank_zero_only
@_catch_inactive
def log_model_summary(self, model: 'pl.LightningModule', max_depth: int=-1) -> None:
    if _NEPTUNE_AVAILABLE:
        from neptune.types import File
    else:
        from neptune.new.types import File
    model_str = str(ModelSummary(model=model, max_depth=max_depth))
    self.run[self._construct_path_with_prefix('model/summary')] = File.from_content(content=model_str, extension='txt')