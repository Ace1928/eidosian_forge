import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
@staticmethod
def get_experiment_dir_name(run_obj: Union[str, Callable, Type]) -> str:
    from ray.tune.experiment import Experiment
    from ray.tune.utils import date_str
    run_identifier = Experiment.get_trainable_name(run_obj)
    if bool(int(os.environ.get('TUNE_DISABLE_DATED_SUBDIR', 0))):
        dir_name = run_identifier
    else:
        dir_name = '{}_{}'.format(run_identifier, date_str())
    return dir_name