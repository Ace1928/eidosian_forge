import copy
import json
import time
import traceback
import uuid
import warnings
from collections import defaultdict, deque
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple, Set
import logging
import os
import ray
from ray.air import ResourceRequest
from ray.air.constants import TIME_THIS_ITER_S
from ray.air.execution import ResourceManager, PlacementGroupResourceManager
from ray.air.execution._internal import RayActorManager, TrackedActor
from ray.train import CheckpointConfig
from ray.train._internal.session import _FutureTrainingResult
from ray.train._internal.storage import StorageContext
from ray.exceptions import RayActorError, RayTaskError
from ray.tune.error import _AbortTrialExecution, _TuneStopTrialError
from ray.tune.execution.class_cache import _ActorClassCache
from ray.tune.execution.experiment_state import (
from ray.tune.experiment.trial import (
from ray.tune.experiment import Experiment
from ray.tune.execution.insufficient_resources_manager import (
from ray.tune.result import (
from ray.tune.result import TRIAL_INFO, STDOUT_FILE, STDERR_FILE
from ray.tune import TuneError
from ray.tune.callback import Callback, CallbackList
from ray.tune.schedulers import FIFOScheduler, TrialScheduler
from ray.tune.stopper import NoopStopper, Stopper
from ray.tune.search import BasicVariantGenerator, SearchAlgorithm
from ray.tune.experiment import Trial
from ray.tune.utils.log import _dedup_logs
from ray.tune.utils.object_cache import _ObjectCache
from ray.tune.utils.resource_updater import _ResourceUpdater
from ray.tune.utils import warn_if_slow, flatten_dict
from ray.tune.utils.log import Verbosity, has_verbosity
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.utils.serialization import TuneFunctionDecoder, TuneFunctionEncoder
from ray.util.annotations import DeveloperAPI, Deprecated
from ray.util.debug import log_once
def restore_from_dir(self) -> List[Trial]:
    """Restore TrialRunner state from local experiment directory.

        This method will restore the trial runner state, the searcher state,
        and the callback states. It will then parse the trial states
        and return them as a list of Trial objects.
        """
    experiment_dir = self._storage.experiment_local_path
    newest_state_path = _find_newest_experiment_checkpoint(experiment_dir)
    if not newest_state_path:
        raise ValueError(f'Tried to resume experiment from directory `{experiment_dir}`, but no experiment checkpoint data was found.')
    logger.warning(f'Attempting to resume experiment from {experiment_dir}. This will ignore any new changes to the specification.')
    logger.info(f'Using the newest experiment state file found within the experiment directory: {Path(newest_state_path).name}')
    with open(newest_state_path, 'r') as f:
        runner_state = json.load(f, cls=TuneFunctionDecoder)
    self.__setstate__(runner_state['runner_data'])
    if self._search_alg.has_checkpoint(experiment_dir):
        self._search_alg.restore_from_dir(experiment_dir)
    if self._callbacks.can_restore(experiment_dir):
        self._callbacks.restore_from_dir(experiment_dir)
    trials = []
    for trial_json_state, trial_runtime_metadata in runner_state['trial_data']:
        trial = Trial.from_json_state(trial_json_state)
        trial.restore_run_metadata(trial_runtime_metadata)
        new_storage = copy.copy(trial.storage)
        new_storage.storage_filesystem = self._storage.storage_filesystem
        new_storage.storage_fs_path = self._storage.storage_fs_path
        new_storage.experiment_dir_name = self._storage.experiment_dir_name
        trial.set_storage(new_storage)
        if not ray.util.client.ray.is_connected():
            trial.init_local_path()
        trials.append(trial)
    return trials