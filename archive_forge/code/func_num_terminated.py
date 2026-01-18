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
def num_terminated(self):
    """Returns the number of terminated (but not errored) trials."""
    return len([t for t in self._experiment_analysis.trials if t.status == Trial.TERMINATED])