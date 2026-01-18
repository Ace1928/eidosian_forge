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
@property
def best_checkpoint(self) -> Checkpoint:
    """Get the checkpoint path of the best trial of the experiment

        The best trial is determined by comparing the last trial results
        using the `metric` and `mode` parameters passed to `tune.run()`.

        If you didn't pass these parameters, use
        `get_best_checkpoint(trial, metric, mode)` instead.

        Returns:
            :class:`Checkpoint <ray.train.Checkpoint>` object.
        """
    if not self.default_metric or not self.default_mode:
        raise ValueError('To fetch the `best_checkpoint`, pass a `metric` and `mode` parameter to `tune.run()`. Alternatively, use the `get_best_checkpoint(trial, metric, mode)` method to set the metric and mode explicitly.')
    best_trial = self.best_trial
    if not best_trial:
        raise ValueError(f'No best trial found. Please check if you specified the correct default metric ({self.default_metric}) and mode ({self.default_mode}).')
    return self.get_best_checkpoint(best_trial, self.default_metric, self.default_mode)