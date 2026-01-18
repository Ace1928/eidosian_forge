import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
def generate_many(self, n: int, seed: Any=None) -> List[Any]:
    """Generate ``n`` randomly chosen values

        :param n: number of random values to generate
        :param seed: random seed, defaults to None
        :return: a list of values
        """
    if seed is not None:
        np.random.seed(seed)
    return [self.generate() for _ in range(n)]