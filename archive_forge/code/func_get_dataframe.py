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
def get_dataframe(self, filter_metric: Optional[str]=None, filter_mode: Optional[str]=None) -> pd.DataFrame:
    """Return dataframe of all trials with their configs and reported results.

        Per default, this returns the last reported results for each trial.

        If ``filter_metric`` and ``filter_mode`` are set, the results from each
        trial are filtered for this metric and mode. For example, if
        ``filter_metric="some_metric"`` and ``filter_mode="max"``, for each trial,
        every received result is checked, and the one where ``some_metric`` is
        maximal is returned.


        Example:

            .. testcode::

                from ray import train
                from ray.train import RunConfig
                from ray.tune import Tuner

                def training_loop_per_worker(config):
                    train.report({"accuracy": 0.8})

                result_grid = Tuner(
                    trainable=training_loop_per_worker,
                    run_config=RunConfig(name="my_tune_run")
                ).fit()

                # Get last reported results per trial
                df = result_grid.get_dataframe()

                # Get best ever reported accuracy per trial
                df = result_grid.get_dataframe(
                    filter_metric="accuracy", filter_mode="max"
                )

            .. testoutput::
                :hide:

                ...

        Args:
            filter_metric: Metric to filter best result for.
            filter_mode: If ``filter_metric`` is given, one of ``["min", "max"]``
                to specify if we should find the minimum or maximum result.

        Returns:
            Pandas DataFrame with each trial as a row and their results as columns.
        """
    return self._experiment_analysis.dataframe(metric=filter_metric, mode=filter_mode)