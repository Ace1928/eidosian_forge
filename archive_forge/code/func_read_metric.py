import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def read_metric(path: str) -> List[int]:
    """
    Read metric file in JSON or Rdump format.
    Return dimensions of entry "inv_metric".
    """
    if path.endswith('.json'):
        with open(path, 'r') as fd:
            metric_dict = json.load(fd)
        if 'inv_metric' in metric_dict:
            dims_np: np.ndarray = np.asarray(metric_dict['inv_metric'])
            return list(dims_np.shape)
        else:
            raise ValueError('metric file {}, bad or missing entry "inv_metric"'.format(path))
    else:
        dims = list(read_rdump_metric(path))
        if dims is None:
            raise ValueError('metric file {}, bad or missing entry "inv_metric"'.format(path))
        return dims