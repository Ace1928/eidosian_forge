import math
import os
from io import StringIO
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP, _TMPDIR
from cmdstanpy.cmdstan_args import Method, SamplerArgs
from cmdstanpy.utils import (
from .metadata import InferenceMetadata
from .runset import RunSet
@property
def step_size(self) -> Optional[np.ndarray]:
    """
        Step size used by sampler for each chain.
        When sampler algorithm 'fixed_param' is specified, step size is None.
        """
    self._assemble_draws()
    return self._step_size if not self._is_fixed_param else None