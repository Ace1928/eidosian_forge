import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def scan_config(fd: TextIO, config_dict: Dict[str, Any], lineno: int) -> int:
    """
    Scan initial stan_csv file comments lines and
    save non-default configuration information to config_dict.
    """
    cur_pos = fd.tell()
    line = fd.readline().strip()
    while len(line) > 0 and line.startswith('#'):
        lineno += 1
        if line.endswith('(Default)'):
            line = line.replace('(Default)', '')
        line = line.lstrip(' #\t')
        key_val = line.split('=')
        if len(key_val) == 2:
            if key_val[0].strip() == 'file' and (not key_val[1].endswith('csv')):
                config_dict['data_file'] = key_val[1].strip()
            elif key_val[0].strip() != 'file':
                raw_val = key_val[1].strip()
                val: Union[int, float, str]
                try:
                    val = int(raw_val)
                except ValueError:
                    try:
                        val = float(raw_val)
                    except ValueError:
                        val = raw_val
                config_dict[key_val[0].strip()] = val
        cur_pos = fd.tell()
        line = fd.readline().strip()
    fd.seek(cur_pos)
    return lineno