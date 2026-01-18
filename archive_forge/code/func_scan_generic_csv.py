import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def scan_generic_csv(path: str) -> Dict[str, Any]:
    """Process laplace stan_csv output file line by line."""
    dict: Dict[str, Any] = {}
    lineno = 0
    with open(path, 'r') as fd:
        lineno = scan_config(fd, dict, lineno)
        lineno = scan_column_names(fd, dict, lineno)
    return dict