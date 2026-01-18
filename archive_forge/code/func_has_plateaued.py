import numpy as np
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper
def has_plateaued(self):
    return len(self._top_values) == self._top and np.std(self._top_values) <= self._std