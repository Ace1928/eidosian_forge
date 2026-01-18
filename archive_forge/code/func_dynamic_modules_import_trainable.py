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
@functools.wraps(trainable)
def dynamic_modules_import_trainable(*args, **kwargs):
    """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
    if is_datasets_available():
        import datasets.load
        dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), '__init__.py')
        spec = importlib.util.spec_from_file_location('datasets_modules', dynamic_modules_path)
        datasets_modules = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = datasets_modules
        spec.loader.exec_module(datasets_modules)
    return trainable(*args, **kwargs)