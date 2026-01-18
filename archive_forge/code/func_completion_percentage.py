import collections
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging
from ray.util.annotations import PublicAPI
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.tune.error import TuneError
def completion_percentage(self) -> float:
    """Returns a progress metric.

        This will not be always finish with 100 since dead trials
        are dropped."""
    if self.finished():
        return 1.0
    return min(self._completed_progress / self._total_work, 1.0)