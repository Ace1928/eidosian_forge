import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def add_trial(self, trial: Trial):
    """Add trial to bracket assuming bracket is not filled.

        At a later iteration, a newly added trial will be given equal
        opportunity to catch up."""
    assert not self.filled(), 'Cannot add trial to filled bracket!'
    self._live_trials[trial] = None
    self._all_trials.append(trial)