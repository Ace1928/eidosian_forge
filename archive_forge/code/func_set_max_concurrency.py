import copy
import logging
import math
from ray import cloudpickle
from typing import Dict, List, Optional, Union
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_list_dict
def set_max_concurrency(self, max_concurrent: int) -> bool:
    self._max_concurrent = max_concurrent
    return True