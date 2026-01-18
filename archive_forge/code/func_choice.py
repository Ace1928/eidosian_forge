import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
@PublicAPI
def choice(categories: Sequence):
    """Sample a categorical value.

    Sampling from ``tune.choice([1, 2])`` is equivalent to sampling from
    ``np.random.choice([1, 2])``

    """
    return Categorical(categories).uniform()