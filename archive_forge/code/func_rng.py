import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@property
def rng(self):
    return self._rng if self._rng is not None else np.random