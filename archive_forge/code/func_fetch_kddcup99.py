import errno
import logging
import os
from gzip import GzipFile
from os.path import exists, join
import joblib
import numpy as np
from ..utils import Bunch, check_random_state
from ..utils import shuffle as shuffle_method
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home
from ._base import (
@validate_params({'subset': [StrOptions({'SA', 'SF', 'http', 'smtp'}), None], 'data_home': [str, os.PathLike, None], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'percent10': ['boolean'], 'download_if_missing': ['boolean'], 'return_X_y': ['boolean'], 'as_frame': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_kddcup99(*, subset=None, data_home=None, shuffle=False, random_state=None, percent10=True, download_if_missing=True, return_X_y=False, as_frame=False):
    """Load the kddcup99 dataset (classification).

    Download it if necessary.

    =================   ====================================
    Classes                                               23
    Samples total                                    4898431
    Dimensionality                                        41
    Features            discrete (int) or continuous (float)
    =================   ====================================

    Read more in the :ref:`User Guide <kddcup99_dataset>`.

    .. versionadded:: 0.18

    Parameters
    ----------
    subset : {'SA', 'SF', 'http', 'smtp'}, default=None
        To return the corresponding classical subsets of kddcup 99.
        If None, return the entire kddcup 99 dataset.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the datasets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

        .. versionadded:: 0.19

    shuffle : bool, default=False
        Whether to shuffle dataset.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset shuffling and for
        selection of abnormal samples if `subset='SA'`. Pass an int for
        reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    percent10 : bool, default=True
        Whether to load only 10 percent of the data.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` object.

        .. versionadded:: 0.20

    as_frame : bool, default=False
        If `True`, returns a pandas Dataframe for the ``data`` and ``target``
        objects in the `Bunch` returned object; `Bunch` return object will also
        have a ``frame`` member.

        .. versionadded:: 0.24

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : {ndarray, dataframe} of shape (494021, 41)
            The data matrix to learn. If `as_frame=True`, `data` will be a
            pandas DataFrame.
        target : {ndarray, series} of shape (494021,)
            The regression target for each sample. If `as_frame=True`, `target`
            will be a pandas Series.
        frame : dataframe of shape (494021, 42)
            Only present when `as_frame=True`. Contains `data` and `target`.
        DESCR : str
            The full description of the dataset.
        feature_names : list
            The names of the dataset columns
        target_names: list
            The names of the target columns

    (data, target) : tuple if ``return_X_y`` is True
        A tuple of two ndarray. The first containing a 2D array of
        shape (n_samples, n_features) with each row representing one
        sample and each column representing the features. The second
        ndarray of shape (n_samples,) containing the target samples.

        .. versionadded:: 0.20
    """
    data_home = get_data_home(data_home=data_home)
    kddcup99 = _fetch_brute_kddcup99(data_home=data_home, percent10=percent10, download_if_missing=download_if_missing)
    data = kddcup99.data
    target = kddcup99.target
    feature_names = kddcup99.feature_names
    target_names = kddcup99.target_names
    if subset == 'SA':
        s = target == b'normal.'
        t = np.logical_not(s)
        normal_samples = data[s, :]
        normal_targets = target[s]
        abnormal_samples = data[t, :]
        abnormal_targets = target[t]
        n_samples_abnormal = abnormal_samples.shape[0]
        random_state = check_random_state(random_state)
        r = random_state.randint(0, n_samples_abnormal, 3377)
        abnormal_samples = abnormal_samples[r]
        abnormal_targets = abnormal_targets[r]
        data = np.r_[normal_samples, abnormal_samples]
        target = np.r_[normal_targets, abnormal_targets]
    if subset == 'SF' or subset == 'http' or subset == 'smtp':
        s = data[:, 11] == 1
        data = np.c_[data[s, :11], data[s, 12:]]
        feature_names = feature_names[:11] + feature_names[12:]
        target = target[s]
        data[:, 0] = np.log((data[:, 0] + 0.1).astype(float, copy=False))
        data[:, 4] = np.log((data[:, 4] + 0.1).astype(float, copy=False))
        data[:, 5] = np.log((data[:, 5] + 0.1).astype(float, copy=False))
        if subset == 'http':
            s = data[:, 2] == b'http'
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[4], feature_names[5]]
        if subset == 'smtp':
            s = data[:, 2] == b'smtp'
            data = data[s]
            target = target[s]
            data = np.c_[data[:, 0], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[4], feature_names[5]]
        if subset == 'SF':
            data = np.c_[data[:, 0], data[:, 2], data[:, 4], data[:, 5]]
            feature_names = [feature_names[0], feature_names[2], feature_names[4], feature_names[5]]
    if shuffle:
        data, target = shuffle_method(data, target, random_state=random_state)
    fdescr = load_descr('kddcup99.rst')
    frame = None
    if as_frame:
        frame, data, target = _convert_data_dataframe('fetch_kddcup99', data, target, feature_names, target_names)
    if return_X_y:
        return (data, target)
    return Bunch(data=data, target=target, frame=frame, target_names=target_names, feature_names=feature_names, DESCR=fdescr)