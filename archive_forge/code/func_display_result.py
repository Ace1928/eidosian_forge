from __future__ import print_function
import collections
import datetime
import numbers
import os
import sys
import textwrap
import time
import warnings
from typing import Any, Callable, Collection, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import ray
from ray._private.dict import flatten_dict
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.experimental.tqdm_ray import safe_print
from ray.air.util.node import _force_on_current_node
from ray.air.constants import EXPR_ERROR_FILE, TRAINING_ITERATION
from ray.tune.callback import Callback
from ray.tune.logger import pretty_print
from ray.tune.result import (
from ray.tune.experiment.trial import DEBUG_PRINT_INTERVAL, Trial, _Location
from ray.tune.trainable import Trainable
from ray.tune.utils import unflattened_lookup
from ray.tune.utils.log import Verbosity, has_verbosity, set_verbosity
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.util.queue import Empty, Queue
from ray.widgets import Template
def display_result(self, trial: Trial, result: Dict, error: bool, done: bool):
    """Display a formatted HTML table of trial progress results.

        Trial progress is only shown if verbosity is set to level 2 or 3.

        Args:
            trial: Trial for which results are to be printed
            result: Result to be printed
            error: True if an error has occurred, False otherwise
            done: True if the trial is finished, False otherwise
        """
    from IPython.display import display, HTML
    self._last_result[trial] = result
    if has_verbosity(Verbosity.V3_TRIAL_DETAILS):
        ignored_keys = {'config', 'hist_stats'}
    elif has_verbosity(Verbosity.V2_TRIAL_NORM):
        ignored_keys = {'config', 'hist_stats', 'trial_id', 'experiment_tag', 'done'} | set(AUTO_RESULT_KEYS)
    else:
        return
    table = self.generate_trial_table(self._last_result, set(result.keys()) - ignored_keys)
    if not self._display_handle:
        self._display_handle = display(HTML(table), display_id=True)
    else:
        self._display_handle.update(HTML(table))