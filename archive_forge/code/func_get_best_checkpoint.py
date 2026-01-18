import copy
import fnmatch
import io
import json
import logging
from numbers import Number
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow.fs
from ray.util.annotations import PublicAPI
from ray.air.constants import (
from ray.train import Checkpoint
from ray.train._internal.storage import (
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.tune.utils.serialization import TuneFunctionDecoder
from ray.tune.utils.util import is_nan_or_inf, is_nan, unflattened_lookup
def get_best_checkpoint(self, trial: Trial, metric: Optional[str]=None, mode: Optional[str]=None) -> Optional[Checkpoint]:
    """Gets best persistent checkpoint path of provided trial.

        Any checkpoints with an associated metric value of ``nan`` will be filtered out.

        Args:
            trial: The log directory of a trial, or a trial instance.
            metric: key of trial info to return, e.g. "mean_accuracy".
                "training_iteration" is used by default if no value was
                passed to ``self.default_metric``.
            mode: One of [min, max]. Defaults to ``self.default_mode``.

        Returns:
            A :class:`Checkpoint <ray.train.Checkpoint>` object
        """
    metric = metric or self.default_metric or TRAINING_ITERATION
    mode = self._validate_mode(mode)
    checkpoints_and_metrics = self._get_trial_checkpoints_with_metric(trial, metric)
    checkpoints_and_metrics = list(filter(lambda x: not is_nan(x[1]), checkpoints_and_metrics))
    if not checkpoints_and_metrics:
        logger.error(f'No checkpoints have been found for trial {trial}.')
        return None
    score_order_factor = -1 if mode == 'min' else 1
    best_checkpoint, _ = max(checkpoints_and_metrics, key=lambda x: score_order_factor * x[1])
    return best_checkpoint