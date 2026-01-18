import copy
import io
import os
import math
import logging
from pathlib import Path
from typing import (
import pyarrow.fs
import ray.cloudpickle as pickle
from ray.util import inspect_serializability
from ray.air._internal.uri_utils import URI
from ray.air._internal.usage import AirEntrypoint
from ray.air.config import RunConfig, ScalingConfig
from ray.train._internal.storage import StorageContext, get_fs_and_path
from ray.tune import Experiment, TuneError, ExperimentAnalysis
from ray.tune.execution.experiment_state import _ResumeConfig
from ray.tune.tune import _Config
from ray.tune.registry import is_function_trainable
from ray.tune.result import _get_defaults_results_dir
from ray.tune.result_grid import ResultGrid
from ray.tune.trainable import Trainable
from ray.tune.tune import run
from ray.tune.tune_config import TuneConfig
from ray.tune.utils import flatten_dict
@classmethod
def setup_create_experiment_checkpoint_dir(cls, trainable: TrainableType, run_config: Optional[RunConfig]) -> Tuple[str, str]:
    """Sets up and creates the local experiment checkpoint dir.
        This is so that the `tuner.pkl` file gets stored in the same directory
        and gets synced with other experiment results.

        Returns:
            Tuple: (experiment_path, experiment_dir_name)
        """
    experiment_dir_name = run_config.name or StorageContext.get_experiment_dir_name(trainable)
    storage_local_path = _get_defaults_results_dir()
    experiment_path = Path(storage_local_path).joinpath(experiment_dir_name).as_posix()
    os.makedirs(experiment_path, exist_ok=True)
    return (experiment_path, experiment_dir_name)