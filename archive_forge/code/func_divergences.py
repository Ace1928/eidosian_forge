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
def divergences(self) -> Optional[np.ndarray]:
    """
        Per-chain total number of post-warmup divergent iterations.
        When sampler algorithm 'fixed_param' is specified, returns None.
        """
    return self._divergences if not self._is_fixed_param else None