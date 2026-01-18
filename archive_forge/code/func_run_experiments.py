import abc
import copy
import datetime
import logging
import os
import signal
import sys
import threading
import time
import warnings
from typing import (
import ray
from ray.air._internal import usage as air_usage
from ray.air._internal.usage import AirEntrypoint
from ray.air.util.node import _force_on_current_node
from ray.train import CheckpointConfig, SyncConfig
from ray.train.constants import RAY_CHDIR_TO_TRIAL_DIR, _DEPRECATED_VALUE
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.callback import Callback
from ray.tune.error import TuneError
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Experiment, _convert_to_experiment_list
from ray.tune.experimental.output import (
from ray.tune.impl.placeholder import create_resolvers_map, inject_placeholders
from ray.tune.logger import TBXLoggerCallback
from ray.tune.progress_reporter import (
from ray.tune.registry import get_trainable_cls, is_function_trainable
from ray.tune.schedulers import (
from ray.tune.schedulers.util import (
from ray.tune.stopper import Stopper
from ray.tune.search import (
from ray.tune.search.util import (
from ray.tune.search.variant_generator import _has_unresolved_values
from ray.tune.trainable import Trainable
from ray.tune.experiment import Trial
from ray.tune.utils.callback import _create_default_callbacks
from ray.tune.utils.log import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.util.annotations import PublicAPI
from ray.util.queue import Queue
@PublicAPI
def run_experiments(experiments: Union[Experiment, Mapping, Sequence[Union[Experiment, Mapping]]], scheduler: Optional[TrialScheduler]=None, verbose: Optional[Union[int, AirVerbosity, Verbosity]]=None, progress_reporter: Optional[ProgressReporter]=None, resume: Union[bool, str]=False, reuse_actors: Optional[bool]=None, raise_on_failed_trial: bool=True, concurrent: bool=True, callbacks: Optional[Sequence[Callback]]=None, _remote: Optional[bool]=None):
    """Runs and blocks until all trials finish.

    Example:
        >>> from ray.tune.experiment import Experiment
        >>> from ray.tune.tune import run_experiments
        >>> def my_func(config): return {"score": 0}
        >>> experiment_spec = Experiment("experiment", my_func) # doctest: +SKIP
        >>> run_experiments(experiments=experiment_spec) # doctest: +SKIP
        >>> experiment_spec = {"experiment": {"run": my_func}} # doctest: +SKIP
        >>> run_experiments(experiments=experiment_spec) # doctest: +SKIP

    Returns:
        List of Trial objects, holding data for each executed trial.

    """
    if _remote is None:
        _remote = ray.util.client.ray.is_connected()
    _ray_auto_init(entrypoint='tune.run_experiments(...)')
    if verbose is None:
        verbose = get_air_verbosity(AirVerbosity.VERBOSE) or Verbosity.V3_TRIAL_DETAILS
    if _remote:
        if get_air_verbosity(verbose) is not None:
            logger.info('[output] This uses the legacy output and progress reporter, as Ray client is not supported by the new engine. For more information, see https://github.com/ray-project/ray/issues/36949')
        remote_run = ray.remote(num_cpus=0)(run_experiments)
        remote_run = _force_on_current_node(remote_run)
        return ray.get(remote_run.remote(experiments, scheduler, verbose, progress_reporter, resume, reuse_actors, raise_on_failed_trial, concurrent, callbacks, _remote=False))
    experiments = _convert_to_experiment_list(experiments)
    if concurrent:
        return run(experiments, verbose=verbose, progress_reporter=progress_reporter, resume=resume, reuse_actors=reuse_actors, raise_on_failed_trial=raise_on_failed_trial, scheduler=scheduler, callbacks=callbacks, _entrypoint=AirEntrypoint.TUNE_RUN_EXPERIMENTS).trials
    else:
        trials = []
        for exp in experiments:
            trials += run(exp, verbose=verbose, progress_reporter=progress_reporter, resume=resume, reuse_actors=reuse_actors, raise_on_failed_trial=raise_on_failed_trial, scheduler=scheduler, callbacks=callbacks, _entrypoint=AirEntrypoint.TUNE_RUN_EXPERIMENTS).trials
        return trials