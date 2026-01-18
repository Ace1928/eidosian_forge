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
def num_draws_sampling(self) -> int:
    """
        Number of sampling (post-warmup) draws per chain, i.e.,
        thinned sampling iterations.
        """
    return int(math.ceil(self._iter_sampling / self._thin))