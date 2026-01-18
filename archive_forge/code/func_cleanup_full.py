import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def cleanup_full(self, tune_controller: 'TuneController'):
    """Cleans up bracket after bracket is completely finished.

        Lets the last trial continue to run until termination condition
        kicks in."""
    for trial in self.current_trials():
        if trial.status == Trial.PAUSED:
            tune_controller.stop_trial(trial)