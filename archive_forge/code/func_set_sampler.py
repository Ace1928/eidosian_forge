import logging
from copy import copy
from inspect import signature
from math import isclose
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import numpy as np
from ray.util.annotations import DeveloperAPI, PublicAPI
def set_sampler(self, sampler, allow_override=False):
    if self.sampler and (not allow_override):
        raise ValueError('You can only choose one sampler for parameter domains. Existing sampler for parameter {}: {}. Tried to add {}'.format(self.__class__.__name__, self.sampler, sampler))
    self.sampler = sampler