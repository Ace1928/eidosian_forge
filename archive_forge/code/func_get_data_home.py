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
def get_data_home(data_home=None) -> str:
    """Return the path of the scikit-learn data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'scikit_learn_data' in the
    user home folder.

    Alternatively, it can be set by the 'SCIKIT_LEARN_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/scikit_learn_data`.

    Returns
    -------
    data_home: str
        The path to scikit-learn data directory.
    """
    if data_home is None:
        data_home = environ.get('SCIKIT_LEARN_DATA', join('~', 'scikit_learn_data'))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home