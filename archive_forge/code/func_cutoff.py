import logging
from typing import Dict, Optional, Union, TYPE_CHECKING
import numpy as np
import pickle
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def cutoff(self, recorded) -> Optional[Union[int, float, complex, np.ndarray]]:
    if not recorded:
        return None
    return np.nanpercentile(list(recorded.values()), (1 - 1 / self.rf) * 100)