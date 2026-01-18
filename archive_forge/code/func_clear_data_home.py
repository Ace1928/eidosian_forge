import csv
import gzip
import hashlib
import os
import shutil
from collections import namedtuple
from importlib import resources
from numbers import Integral
from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from ..preprocessing import scale
from ..utils import Bunch, check_pandas_support, check_random_state
from ..utils._param_validation import Interval, StrOptions, validate_params
@validate_params({'data_home': [str, os.PathLike, None]}, prefer_skip_nested_validation=True)
def clear_data_home(data_home=None):
    """Delete all the content of the data home cache.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/scikit_learn_data`.

    Examples
    --------
    >>> from sklearn.datasets import clear_data_home
    >>> clear_data_home()  # doctest: +SKIP
    """
    data_home = get_data_home(data_home)
    shutil.rmtree(data_home)