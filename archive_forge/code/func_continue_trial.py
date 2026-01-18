import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def continue_trial(self, trial: Trial) -> bool:
    result = self._live_trials[trial]
    if not self.stop_last_trials and self._halves == 0:
        return True
    elif self._get_result_time(result) < self._cumul_r:
        logger.debug(f"Continuing trial {trial} as it hasn't reached the time threshold {self._cumul_r}, yet.")
        return True
    return False