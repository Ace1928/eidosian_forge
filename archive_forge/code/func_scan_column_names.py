import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def scan_column_names(fd: TextIO, config_dict: MutableMapping[str, Any], lineno: int) -> int:
    """
    Process columns header, add to config_dict as 'column_names'
    """
    line = fd.readline().strip()
    lineno += 1
    config_dict['raw_header'] = line.strip()
    names = line.split(',')
    config_dict['column_names'] = tuple(munge_varnames(names))
    return lineno