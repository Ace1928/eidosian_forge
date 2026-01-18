from typing import (
import numpy as np
from gym.utils import seeding
@property
def np_random(self) -> np.random.Generator:
    """Lazily seed the PRNG since this is expensive and only needed if sampling from this space."""
    if self._np_random is None:
        self.seed()
    return self._np_random