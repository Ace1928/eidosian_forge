import json
import math
import re
from typing import Any, Dict, List, MutableMapping, Optional, TextIO, Union
import numpy as np
import pandas as pd
from cmdstanpy import _CMDSTAN_SAMPLING, _CMDSTAN_THIN, _CMDSTAN_WARMUP
def parse_rdump_value(rhs: str) -> Union[int, float, np.ndarray]:
    """Process right hand side of Rdump variable assignment statement.
    Value is either scalar, vector, or multi-dim structure.
    Use regex to capture structure values, dimensions.
    """
    pat = re.compile('structure\\(\\s*c\\((?P<vals>[^)]*)\\)(,\\s*\\.Dim\\s*=\\s*c\\s*\\((?P<dims>[^)]*)\\s*\\))?\\)')
    val: Union[int, float, np.ndarray]
    try:
        if rhs.startswith('structure'):
            parse = pat.match(rhs)
            if parse is None or parse.group('vals') is None:
                raise ValueError(rhs)
            vals = [float(v) for v in parse.group('vals').split(',')]
            val = np.array(vals, order='F')
            if parse.group('dims') is not None:
                dims = [int(v) for v in parse.group('dims').split(',')]
                val = np.array(vals).reshape(dims, order='F')
        elif rhs.startswith('c(') and rhs.endswith(')'):
            val = np.array([float(item) for item in rhs[2:-1].split(',')])
        elif '.' in rhs or 'e' in rhs:
            val = float(rhs)
        else:
            val = int(rhs)
    except TypeError as e:
        raise ValueError('bad value in Rdump file: {}'.format(rhs)) from e
    return val