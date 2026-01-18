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
def save_to_dir(self):
    """Save TuneController state to the local experiment directory.

        This includes:
        - trial states
        - TuneController internal state (all the serializable attributes)
        - the searcher state
        - the callback states
        """
    experiment_dir = self._storage.experiment_local_path
    runner_state = {'trial_data': list(self._get_trial_checkpoints().values()), 'runner_data': self.__getstate__(), 'stats': {'start_time': self._start_time, 'timestamp': self._last_checkpoint_time}}
    tmp_file_name = os.path.join(experiment_dir, f'.tmp_experiment_state_{uuid.uuid4()}')
    with open(tmp_file_name, 'w') as f:
        json.dump(runner_state, f, indent=2, cls=TuneFunctionEncoder)
    os.replace(tmp_file_name, os.path.join(experiment_dir, self.experiment_state_file_name))
    self._search_alg.save_to_dir(experiment_dir, session_str=self._session_str)
    self._callbacks.save_to_dir(experiment_dir, session_str=self._session_str)