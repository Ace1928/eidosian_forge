import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
def on_log(self, args, state, control, model=None, logs=None, **kwargs):
    if not self._initialized:
        self.setup(args, state, model)
    if state.is_world_process_zero:
        from dvclive.plots import Metric
        from dvclive.utils import standardize_metric_name
        for key, value in logs.items():
            if Metric.could_log(value):
                self.live.log_metric(standardize_metric_name(key, 'dvclive.huggingface'), value)
            else:
                logger.warning(f'''Trainer is attempting to log a value of "{value}" of type {type(value)} for key "{key}" as a scalar. This invocation of DVCLive's Live.log_metric() is incorrect so we dropped this attribute.''')
        self.live.next_step()