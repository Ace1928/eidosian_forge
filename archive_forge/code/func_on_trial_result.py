import time
import logging
import pickle
import functools
import warnings
from packaging import version
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from ray.air.constants import TRAINING_ITERATION
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search.sample import (
from ray.tune.search import (
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils.util import flatten_dict, unflatten_dict, validate_warmstart
def on_trial_result(self, trial_id: str, result: Dict):
    if isinstance(self.metric, list):
        return
    if trial_id in self._completed_trials:
        logger.warning(f'Received additional result for trial {trial_id}, but it already finished. Result: {result}')
        return
    metric = result[self.metric]
    step = result[TRAINING_ITERATION]
    ot_trial = self._ot_trials[trial_id]
    ot_trial.report(metric, step)