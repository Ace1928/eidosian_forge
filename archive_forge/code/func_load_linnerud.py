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
@validate_params({'return_X_y': ['boolean'], 'as_frame': ['boolean']}, prefer_skip_nested_validation=True)
def load_linnerud(*, return_X_y=False, as_frame=False):
    """Load and return the physical exercise Linnerud dataset.

    This dataset is suitable for multi-output regression tasks.

    ==============   ============================
    Samples total    20
    Dimensionality   3 (for both data and target)
    Features         integer
    Targets          integer
    ==============   ============================

    Read more in the :ref:`User Guide <linnerrud_dataset>`.

    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.

        .. versionadded:: 0.18

    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.

        .. versionadded:: 0.23

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (20, 3)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, dataframe} of shape (20, 3)
            The regression targets. If `as_frame=True`, `target` will be
            a pandas DataFrame.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.
        frame: DataFrame of shape (20, 6)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.

            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
        data_filename: str
            The path to the location of the data.
        target_filename: str
            The path to the location of the target.

            .. versionadded:: 0.20

    (data, target) : tuple if ``return_X_y`` is True
        Returns a tuple of two ndarrays or dataframe of shape
        `(20, 3)`. Each row represents one sample and each column represents the
        features in `X` and a target in `y` of a given sample.

        .. versionadded:: 0.18
    """
    data_filename = 'linnerud_exercise.csv'
    target_filename = 'linnerud_physiological.csv'
    data_module_path = resources.files(DATA_MODULE)
    data_path = data_module_path / data_filename
    with data_path.open('r', encoding='utf-8') as f:
        header_exercise = f.readline().split()
        f.seek(0)
        data_exercise = np.loadtxt(f, skiprows=1)
    target_path = data_module_path / target_filename
    with target_path.open('r', encoding='utf-8') as f:
        header_physiological = f.readline().split()
        f.seek(0)
        data_physiological = np.loadtxt(f, skiprows=1)
    fdescr = load_descr('linnerud.rst')
    frame = None
    if as_frame:
        frame, data_exercise, data_physiological = _convert_data_dataframe('load_linnerud', data_exercise, data_physiological, header_exercise, header_physiological)
    if return_X_y:
        return (data_exercise, data_physiological)
    return Bunch(data=data_exercise, feature_names=header_exercise, target=data_physiological, target_names=header_physiological, frame=frame, DESCR=fdescr, data_filename=data_filename, target_filename=target_filename, data_module=DATA_MODULE)