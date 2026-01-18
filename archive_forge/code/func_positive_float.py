import os
from enum import Enum, auto
from time import time
from typing import Any, Dict, List, Mapping, Optional, Union
import numpy as np
from numpy.random import default_rng
from cmdstanpy import _TMPDIR
from cmdstanpy.utils import (
def positive_float(value: Any, name: str) -> None:
    if value is not None:
        if isinstance(value, (int, float, np.floating)):
            if value <= 0:
                raise ValueError(f'{name} must be greater than 0')
        else:
            raise ValueError(f'{name} must be of type float')