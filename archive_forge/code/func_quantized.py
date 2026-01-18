import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
def quantized(self, q: int):
    new = copy(self)
    new.set_sampler(Quantized(new.get_sampler(), q), allow_override=True)
    return new