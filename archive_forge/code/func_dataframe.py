import copy
import fnmatch
import io
import json
import logging
from numbers import Number
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pyarrow.fs
from ray.util.annotations import PublicAPI
from ray.air.constants import (
from ray.train import Checkpoint
from ray.train._internal.storage import (
from ray.tune.execution.tune_controller import TuneController
from ray.tune.experiment import Trial
from ray.tune.result import (
from ray.tune.utils import flatten_dict
from ray.tune.utils.serialization import TuneFunctionDecoder
from ray.tune.utils.util import is_nan_or_inf, is_nan, unflattened_lookup
def dataframe(self, metric: Optional[str]=None, mode: Optional[str]=None) -> DataFrame:
    """Returns a pandas.DataFrame object constructed from the trials.

        This function will look through all observed results of each trial
        and return the one corresponding to the passed ``metric`` and
        ``mode``: If ``mode=min``, it returns the result with the lowest
        *ever* observed ``metric`` for this trial (this is not necessarily
        the last)! For ``mode=max``, it's the highest, respectively. If
        ``metric=None`` or ``mode=None``, the last result will be returned.

        Args:
            metric: Key for trial info to order on. If None, uses last result.
            mode: One of [None, "min", "max"].

        Returns:
            pd.DataFrame: Constructed from a result dict of each trial.
        """
    if mode and mode not in ['min', 'max']:
        raise ValueError('If set, `mode` has to be one of [min, max]')
    if mode and (not metric):
        raise ValueError("If a `mode` is passed to `ExperimentAnalysis.dataframe(), you'll also have to pass a `metric`!")
    rows = self._retrieve_rows(metric=metric, mode=mode)
    all_configs = self.get_all_configs(prefix=True)
    for path, config in all_configs.items():
        if path in rows:
            rows[path].update(config)
            rows[path].update(logdir=path)
    return pd.DataFrame(list(rows.values()))