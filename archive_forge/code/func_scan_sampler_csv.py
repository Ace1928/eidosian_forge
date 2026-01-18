import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def scan_sampler_csv(path: str, is_fixed_param: bool=False) -> Dict[str, Any]:
    """Process sampler stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        try:
            lineno = scan_config(fd, dict, lineno)
            lineno = scan_column_names(fd, dict, lineno)
            if not is_fixed_param:
                lineno = scan_warmup_iters(fd, dict, lineno)
                lineno = scan_hmc_params(fd, dict, lineno)
            lineno = scan_sampling_iters(fd, dict, lineno, is_fixed_param)
        except ValueError as e:
            raise ValueError('Error in reading csv file: ' + path) from e
    return dict