import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
def loguniform(self, base: float=10):
    if not self.lower > 0:
        raise ValueError(f'LogUniform requires a lower bound greater than 0.Got: {self.lower}. Did you pass a variable that has been log-transformed? If so, pass the non-transformed value instead.')
    if not 0 < self.upper < float('inf'):
        raise ValueError(f'LogUniform requires a upper bound greater than 0. Got: {self.lower}. Did you pass a variable that has been log-transformed? If so, pass the non-transformed value instead.')
    new = copy(self)
    new.set_sampler(self._LogUniform(base))
    return new