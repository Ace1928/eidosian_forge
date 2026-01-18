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
@property
def filesystem(self) -> pyarrow.fs.FileSystem:
    """Return the filesystem that can be used to access the experiment path.

        Returns:
            pyarrow.fs.FileSystem implementation.
        """
    return self._experiment_analysis._fs