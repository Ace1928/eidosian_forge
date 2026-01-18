from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method, OptimizeArgs
from cmdstanpy.utils import get_logger, scan_optimize_csv
from .metadata import InferenceMetadata
from .runset import RunSet
@property
def optimized_params_dict(self) -> Dict[str, np.float64]:
    """
        Returns all estimates from the optimizer, including `lp__` as a
        Python Dict.  Only returns estimate from final iteration.
        """
    if not self.runset._check_retcodes():
        get_logger().warning('Invalid estimate, optimization failed to converge.')
    return OrderedDict(zip(self.column_names, self._mle))