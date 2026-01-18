import os
import pandas as pd
import pyarrow
from typing import Optional, Union
from ray.air.result import Result
from ray.cloudpickle import cloudpickle
from ray.exceptions import RayTaskError
from ray.tune.analysis import ExperimentAnalysis
from ray.tune.error import TuneError
from ray.tune.experiment import Trial
from ray.util import PublicAPI
def get_best_result(self, metric: Optional[str]=None, mode: Optional[str]=None, scope: str='last', filter_nan_and_inf: bool=True) -> Result:
    """Get the best result from all the trials run.

        Args:
            metric: Key for trial info to order on. Defaults to
                the metric specified in your Tuner's ``TuneConfig``.
            mode: One of [min, max]. Defaults to the mode specified
                in your Tuner's ``TuneConfig``.
            scope: One of [all, last, avg, last-5-avg, last-10-avg].
                If `scope=last`, only look at each trial's final step for
                `metric`, and compare across trials based on `mode=[min,max]`.
                If `scope=avg`, consider the simple average over all steps
                for `metric` and compare across trials based on
                `mode=[min,max]`. If `scope=last-5-avg` or `scope=last-10-avg`,
                consider the simple average over the last 5 or 10 steps for
                `metric` and compare across trials based on `mode=[min,max]`.
                If `scope=all`, find each trial's min/max score for `metric`
                based on `mode`, and compare trials based on `mode=[min,max]`.
            filter_nan_and_inf: If True (default), NaN or infinite
                values are disregarded and these trials are never selected as
                the best trial.
        """
    if len(self._experiment_analysis.trials) == 1:
        return self._trial_to_result(self._experiment_analysis.trials[0])
    if not metric and (not self._experiment_analysis.default_metric):
        raise ValueError('No metric is provided. Either pass in a `metric` arg to `get_best_result` or specify a metric in the `TuneConfig` of your `Tuner`.')
    if not mode and (not self._experiment_analysis.default_mode):
        raise ValueError('No mode is provided. Either pass in a `mode` arg to `get_best_result` or specify a mode in the `TuneConfig` of your `Tuner`.')
    best_trial = self._experiment_analysis.get_best_trial(metric=metric, mode=mode, scope=scope, filter_nan_and_inf=filter_nan_and_inf)
    if not best_trial:
        error_msg = f'No best trial found for the given metric: {metric or self._experiment_analysis.default_metric}. This means that no trial has reported this metric'
        error_msg += ', or all values reported for this metric are NaN. To not ignore NaN values, you can set the `filter_nan_and_inf` arg to False.' if filter_nan_and_inf else '.'
        raise RuntimeError(error_msg)
    return self._trial_to_result(best_trial)