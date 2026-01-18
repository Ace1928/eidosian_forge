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
def get_reporting_integration_callbacks(report_to):
    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(f'{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported.')
    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]