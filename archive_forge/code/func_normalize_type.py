from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Sequence, Union
import numpy as np
def normalize_type(self, value):
    if np.isscalar(value):
        raise TypeError('Expected array, got scalar')
    return np.asarray(value, dtype=self.dtype)