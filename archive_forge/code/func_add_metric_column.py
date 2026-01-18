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
def add_metric_column(self, metric: str, representation: Optional[str]=None):
    """Adds a metric to the existing columns.

        Args:
            metric: Metric to add. This must be a metric being returned
                in training step results.
            representation: Representation to use in table. Defaults to
                `metric`.
        """
    self._metrics_override = True
    if metric in self._metric_columns:
        raise ValueError('Column {} already exists.'.format(metric))
    if isinstance(self._metric_columns, MutableMapping):
        representation = representation or metric
        self._metric_columns[metric] = representation
    else:
        if representation is not None and representation != metric:
            raise ValueError('`representation` cannot differ from `metric` if this reporter was initialized with a list of metric columns.')
        self._metric_columns.append(metric)