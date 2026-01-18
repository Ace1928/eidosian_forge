import os
import inspect
import warnings
import colorsys
from contextlib import contextmanager
from urllib.request import urlopen, urlretrieve
from types import ModuleType
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
from matplotlib.cbook import normalize_kwargs
from seaborn._core.typing import deprecated
from seaborn.external.version import Version
from seaborn.external.appdirs import user_cache_dir
def remove_na(vector):
    """Helper method for removing null values from data vectors.

    Parameters
    ----------
    vector : vector object
        Must implement boolean masking with [] subscript syntax.

    Returns
    -------
    clean_clean : same type as ``vector``
        Vector of data with null values removed. May be a copy or a view.

    """
    return vector[pd.notnull(vector)]