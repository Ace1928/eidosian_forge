import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def scan_sampling_iters(fd: TextIO, config_dict: Dict[str, Any], lineno: int, is_fixed_param: bool) -> int:
    """
    Parse sampling iteration, save number of iterations to config_dict.
    Also save number of divergences, max_treedepth hits
    """
    draws_found = 0
    num_cols = len(config_dict['column_names'])
    if not is_fixed_param:
        idx_divergent = config_dict['column_names'].index('divergent__')
        idx_treedepth = config_dict['column_names'].index('treedepth__')
        max_treedepth = config_dict['max_depth']
        ct_divergences = 0
        ct_max_treedepth = 0
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and (not line.startswith('#')):
        lineno += 1
        draws_found += 1
        data = line.split(',')
        if len(data) != num_cols:
            raise ValueError('line {}: bad draw, expecting {} items, found {}\n'.format(lineno, num_cols, len(line.split(','))) + 'This error could be caused by running out of disk space.\nTry clearing up TEMP or setting output_dir to a path on another drive.')
        cur_pos = fd.tell()
        line = fd.readline().strip()
        if not is_fixed_param:
            ct_divergences += int(data[idx_divergent])
            if int(data[idx_treedepth]) == max_treedepth:
                ct_max_treedepth += 1
    fd.seek(cur_pos)
    config_dict['draws_sampling'] = draws_found
    if not is_fixed_param:
        config_dict['ct_divergences'] = ct_divergences
        config_dict['ct_max_treedepth'] = ct_max_treedepth
    return lineno