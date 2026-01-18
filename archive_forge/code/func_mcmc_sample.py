from collections import Counter
from typing import (
import numpy as np
import pandas as pd
from cmdstanpy.cmdstan_args import Method
from cmdstanpy.utils import build_xarray_data, flatten_chains, get_logger
from cmdstanpy.utils.stancsv import scan_generic_csv
from .mcmc import CmdStanMCMC
from .metadata import InferenceMetadata
from .mle import CmdStanMLE
from .runset import RunSet
from .vb import CmdStanVB
@property
def mcmc_sample(self) -> Union[CmdStanMCMC, CmdStanMLE, CmdStanVB]:
    get_logger().warning('Property `mcmc_sample` is deprecated, use `previous_fit` instead')
    return self.previous_fit