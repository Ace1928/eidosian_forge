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
def stop_trial(self, trial):
    """The canonical implementation of stopping a trial.

        Trials may be in any external status when this function is called.
        If trial is in state PENDING or PAUSED, calls `on_trial_remove` for
        scheduler and `on_trial_complete()` for search_alg.
        If trial is in state RUNNING, calls `on_trial_complete` for scheduler
        and search_alg if RUNNING. Caller to ensure that there is no
        outstanding future to be handled for the trial. If there is, the future
        would be discarded.
        """
    try:
        if trial.status in [Trial.ERROR, Trial.TERMINATED]:
            return
        elif trial.status in [Trial.PENDING, Trial.PAUSED]:
            self._scheduler_alg.on_trial_remove(self, trial)
            self._search_alg.on_trial_complete(trial.trial_id)
        elif trial.status is Trial.RUNNING:
            self._scheduler_alg.on_trial_complete(self, trial, flatten_dict(trial.last_result))
            self._search_alg.on_trial_complete(trial.trial_id, result=flatten_dict(trial.last_result))
        self._callbacks.on_trial_complete(iteration=self._iteration, trials=self._trials, trial=trial)
        self._schedule_graceful_trial_stop(trial)
        self._live_trials.discard(trial)
    except Exception as e:
        logger.exception('Trial %s: Error stopping trial.', trial)
        if self._fail_fast == self.RAISE:
            raise
        if isinstance(e, TuneError):
            self._process_trial_failure(trial, exception=e)
        else:
            self._process_trial_failure(trial, _TuneStopTrialError(traceback.format_exc()))