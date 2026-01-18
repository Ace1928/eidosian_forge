from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method, OptimizeArgs
from cmdstanpy.utils import get_logger, scan_optimize_csv
from .metadata import InferenceMetadata
from .runset import RunSet
@property
def optimized_params_pd(self) -> pd.DataFrame:
    """
        Returns all final estimates from the optimizer as a pandas.DataFrame
        which contains all optimizer outputs, i.e., the value for `lp__`
        as well as all Stan program variables.
        """
    if not self.runset._check_retcodes():
        get_logger().warning('Invalid estimate, optimization failed to converge.')
    return pd.DataFrame([self._mle], columns=self.column_names)