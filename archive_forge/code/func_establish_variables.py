import copy
from textwrap import dedent
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import utils
from . import algorithms as algo
from .axisgrid import FacetGrid, _facet_docs
def establish_variables(self, data, **kws):
    """Extract variables from data or use directly."""
    self.data = data
    any_strings = any([isinstance(v, str) for v in kws.values()])
    if any_strings and data is None:
        raise ValueError('Must pass `data` if using named variables.')
    for var, val in kws.items():
        if isinstance(val, str):
            vector = data[val]
        elif isinstance(val, list):
            vector = np.asarray(val)
        else:
            vector = val
        if vector is not None and vector.shape != (1,):
            vector = np.squeeze(vector)
        if np.ndim(vector) > 1:
            err = 'regplot inputs must be 1d'
            raise ValueError(err)
        setattr(self, var, vector)