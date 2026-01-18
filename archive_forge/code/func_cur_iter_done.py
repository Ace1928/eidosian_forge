import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def cur_iter_done(self) -> bool:
    """Checks if all iterations have completed.

        TODO(rliaw): also check that `t.iterations == self._r`"""
    return all((self._get_result_time(result) >= self._cumul_r for result in self._live_trials.values()))