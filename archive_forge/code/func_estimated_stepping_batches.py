import logging
import math
import os
import warnings
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Dict, Generator, Iterable, List, Optional, Union
from weakref import proxy
import torch
from torch.optim import Optimizer
import pytorch_lightning as pl
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback, Checkpoint, EarlyStopping, ProgressBar
from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.utilities import _log_hyperparams
from pytorch_lightning.loops import _PredictionLoop, _TrainingEpochLoop
from pytorch_lightning.loops.evaluation_loop import _EvaluationLoop
from pytorch_lightning.loops.fit_loop import _FitLoop
from pytorch_lightning.loops.utilities import _parse_loop_limits, _reset_progress
from pytorch_lightning.plugins import _PLUGIN_INPUT, Precision
from pytorch_lightning.profilers import Profiler
from pytorch_lightning.strategies import ParallelStrategy, Strategy
from pytorch_lightning.trainer import call, setup
from pytorch_lightning.trainer.configuration_validator import _verify_loop_configurations
from pytorch_lightning.trainer.connectors.accelerator_connector import (
from pytorch_lightning.trainer.connectors.callback_connector import _CallbackConnector
from pytorch_lightning.trainer.connectors.checkpoint_connector import _CheckpointConnector
from pytorch_lightning.trainer.connectors.data_connector import _DataConnector
from pytorch_lightning.trainer.connectors.logger_connector import _LoggerConnector
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT, _PBAR_DICT, _ResultCollection
from pytorch_lightning.trainer.connectors.signal_connector import _SignalConnector
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerState, TrainerStatus
from pytorch_lightning.utilities import GradClipAlgorithmType, parsing
from pytorch_lightning.utilities.argparse import _defaults_from_env_vars
from pytorch_lightning.utilities.compile import _maybe_unwrap_optimized, _verify_strategy_supports_compile
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import (
from pytorch_lightning.utilities.warnings import PossibleUserWarning
@property
def estimated_stepping_batches(self) -> Union[int, float]:
    """The estimated number of batches that will ``optimizer.step()`` during training.

        This accounts for gradient accumulation and the current trainer configuration. This might sets up your training
        dataloader if hadn't been set up already.

        .. code-block:: python

            def configure_optimizers(self):
                optimizer = ...
                stepping_batches = self.trainer.estimated_stepping_batches
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches)
                return [optimizer], [scheduler]

        Raises:
            MisconfigurationException:
                If estimated stepping batches cannot be computed due to different `accumulate_grad_batches`
                at different epochs.

        """
    if self.max_epochs == -1:
        return float('inf') if self.max_steps == -1 else self.max_steps
    if self.train_dataloader is None:
        rank_zero_info('Loading `train_dataloader` to estimate number of stepping batches.')
        self.fit_loop.setup_data()
    total_batches = self.num_training_batches
    if total_batches == float('inf'):
        return self.max_steps
    assert self.max_epochs is not None
    max_estimated_steps = math.ceil(total_batches / self.accumulate_grad_batches) * max(self.max_epochs, 1)
    max_estimated_steps = min(max_estimated_steps, self.max_steps) if self.max_steps != -1 else max_estimated_steps
    return max_estimated_steps