import gzip
import hashlib
import json
import os
import shutil
import time
from contextlib import closing
from functools import wraps
from os.path import join
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from warnings import warn
import numpy as np
from ..utils import (
from ..utils._param_validation import (
from . import get_data_home
from ._arff_parser import load_arff_from_gzip_file
@validate_params({'name': [str, None], 'version': [Interval(Integral, 1, None, closed='left'), StrOptions({'active'})], 'data_id': [Interval(Integral, 1, None, closed='left'), None], 'data_home': [str, os.PathLike, None], 'target_column': [str, list, None], 'cache': [bool], 'return_X_y': [bool], 'as_frame': [bool, StrOptions({'auto'})], 'n_retries': [Interval(Integral, 1, None, closed='left')], 'delay': [Interval(Real, 0, None, closed='right')], 'parser': [StrOptions({'auto', 'pandas', 'liac-arff'})], 'read_csv_kwargs': [dict, None]}, prefer_skip_nested_validation=True)
def fetch_openml(name: Optional[str]=None, *, version: Union[str, int]='active', data_id: Optional[int]=None, data_home: Optional[Union[str, os.PathLike]]=None, target_column: Optional[Union[str, List]]='default-target', cache: bool=True, return_X_y: bool=False, as_frame: Union[str, bool]='auto', n_retries: int=3, delay: float=1.0, parser: str='auto', read_csv_kwargs: Optional[Dict]=None):
    """Fetch dataset from openml by name or dataset id.

    Datasets are uniquely identified by either an integer ID or by a
    combination of name and version (i.e. there might be multiple
    versions of the 'iris' dataset). Please give either name or data_id
    (not both). In case a name is given, a version can also be
    provided.

    Read more in the :ref:`User Guide <openml>`.

    .. versionadded:: 0.20

    .. note:: EXPERIMENTAL

        The API is experimental (particularly the return value structure),
        and might have small backward-incompatible changes without notice
        or warning in future releases.

    Parameters
    ----------
    name : str, default=None
        String identifier of the dataset. Note that OpenML can have multiple
        datasets with the same name.

    version : int or 'active', default='active'
        Version of the dataset. Can only be provided if also ``name`` is given.
        If 'active' the oldest version that's still active is used. Since
        there may be more than one active version of a dataset, and those
        versions may fundamentally be different from one another, setting an
        exact version is highly recommended.

    data_id : int, default=None
        OpenML ID of the dataset. The most specific way of retrieving a
        dataset. If data_id is not given, name (and potential version) are
        used to obtain a dataset.

    data_home : str or path-like, default=None
        Specify another download and cache folder for the data sets. By default
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    target_column : str, list or None, default='default-target'
        Specify the column name in the data to use as target. If
        'default-target', the standard target column a stored on the server
        is used. If ``None``, all columns are returned as data and the
        target is ``None``. If list (of strings), all columns with these names
        are returned as multi-target (Note: not all scikit-learn classifiers
        can handle all types of multi-output combinations).

    cache : bool, default=True
        Whether to cache the downloaded datasets into `data_home`.

    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object. See
        below for more information about the `data` and `target` objects.

    as_frame : bool or 'auto', default='auto'
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric, string or categorical). The target is
        a pandas DataFrame or Series depending on the number of target_columns.
        The Bunch will contain a ``frame`` attribute with the target and the
        data. If ``return_X_y`` is True, then ``(data, target)`` will be pandas
        DataFrames or Series as describe above.

        If `as_frame` is 'auto', the data and target will be converted to
        DataFrame or Series as if `as_frame` is set to True, unless the dataset
        is stored in sparse format.

        If `as_frame` is False, the data and target will be NumPy arrays and
        the `data` will only contain numerical values when `parser="liac-arff"`
        where the categories are provided in the attribute `categories` of the
        `Bunch` instance. When `parser="pandas"`, no ordinal encoding is made.

        .. versionchanged:: 0.24
           The default value of `as_frame` changed from `False` to `'auto'`
           in 0.24.

    n_retries : int, default=3
        Number of retries when HTTP errors or network timeouts are encountered.
        Error with status code 412 won't be retried as they represent OpenML
        generic errors.

    delay : float, default=1.0
        Number of seconds between retries.

    parser : {"auto", "pandas", "liac-arff"}, default="auto"
        Parser used to load the ARFF file. Two parsers are implemented:

        - `"pandas"`: this is the most efficient parser. However, it requires
          pandas to be installed and can only open dense datasets.
        - `"liac-arff"`: this is a pure Python ARFF parser that is much less
          memory- and CPU-efficient. It deals with sparse ARFF datasets.

        If `"auto"`, the parser is chosen automatically such that `"liac-arff"`
        is selected for sparse ARFF datasets, otherwise `"pandas"` is selected.

        .. versionadded:: 1.2
        .. versionchanged:: 1.4
           The default value of `parser` changes from `"liac-arff"` to
           `"auto"`.

    read_csv_kwargs : dict, default=None
        Keyword arguments passed to :func:`pandas.read_csv` when loading the data
        from a ARFF file and using the pandas parser. It can allow to
        overwrite some default parameters.

        .. versionadded:: 1.3

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : np.array, scipy.sparse.csr_matrix of floats, or pandas DataFrame
            The feature matrix. Categorical features are encoded as ordinals.
        target : np.array, pandas Series or DataFrame
            The regression target or classification labels, if applicable.
            Dtype is float if numeric, and object if categorical. If
            ``as_frame`` is True, ``target`` is a pandas object.
        DESCR : str
            The full description of the dataset.
        feature_names : list
            The names of the dataset columns.
        target_names: list
            The names of the target columns.

        .. versionadded:: 0.22

        categories : dict or None
            Maps each categorical feature name to a list of values, such
            that the value encoded as i is ith in the list. If ``as_frame``
            is True, this is None.
        details : dict
            More metadata from OpenML.
        frame : pandas DataFrame
            Only present when `as_frame=True`. DataFrame with ``data`` and
            ``target``.

    (data, target) : tuple if ``return_X_y`` is True

        .. note:: EXPERIMENTAL

            This interface is **experimental** and subsequent releases may
            change attributes without notice (although there should only be
            minor changes to ``data`` and ``target``).

        Missing values in the 'data' are represented as NaN's. Missing values
        in 'target' are represented as NaN's (numerical target) or None
        (categorical target).

    Notes
    -----
    The `"pandas"` and `"liac-arff"` parsers can lead to different data types
    in the output. The notable differences are the following:

    - The `"liac-arff"` parser always encodes categorical features as `str` objects.
      To the contrary, the `"pandas"` parser instead infers the type while
      reading and numerical categories will be casted into integers whenever
      possible.
    - The `"liac-arff"` parser uses float64 to encode numerical features
      tagged as 'REAL' and 'NUMERICAL' in the metadata. The `"pandas"`
      parser instead infers if these numerical features corresponds
      to integers and uses panda's Integer extension dtype.
    - In particular, classification datasets with integer categories are
      typically loaded as such `(0, 1, ...)` with the `"pandas"` parser while
      `"liac-arff"` will force the use of string encoded class labels such as
      `"0"`, `"1"` and so on.
    - The `"pandas"` parser will not strip single quotes - i.e. `'` - from
      string columns. For instance, a string `'my string'` will be kept as is
      while the `"liac-arff"` parser will strip the single quotes. For
      categorical columns, the single quotes are stripped from the values.

    In addition, when `as_frame=False` is used, the `"liac-arff"` parser
    returns ordinally encoded data where the categories are provided in the
    attribute `categories` of the `Bunch` instance. Instead, `"pandas"` returns
    a NumPy array were the categories are not encoded.

    Examples
    --------
    >>> from sklearn.datasets import fetch_openml
    >>> adult = fetch_openml("adult", version=2)  # doctest: +SKIP
    >>> adult.frame.info()  # doctest: +SKIP
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 48842 entries, 0 to 48841
    Data columns (total 15 columns):
     #   Column          Non-Null Count  Dtype
    ---  ------          --------------  -----
     0   age             48842 non-null  int64
     1   workclass       46043 non-null  category
     2   fnlwgt          48842 non-null  int64
     3   education       48842 non-null  category
     4   education-num   48842 non-null  int64
     5   marital-status  48842 non-null  category
     6   occupation      46033 non-null  category
     7   relationship    48842 non-null  category
     8   race            48842 non-null  category
     9   sex             48842 non-null  category
     10  capital-gain    48842 non-null  int64
     11  capital-loss    48842 non-null  int64
     12  hours-per-week  48842 non-null  int64
     13  native-country  47985 non-null  category
     14  class           48842 non-null  category
    dtypes: category(9), int64(6)
    memory usage: 2.7 MB
    """
    if cache is False:
        data_home = None
    else:
        data_home = get_data_home(data_home=data_home)
        data_home = join(str(data_home), 'openml')
    if name is not None:
        name = name.lower()
        if data_id is not None:
            raise ValueError('Dataset data_id={} and name={} passed, but you can only specify a numeric data_id or a name, not both.'.format(data_id, name))
        data_info = _get_data_info_by_name(name, version, data_home, n_retries=n_retries, delay=delay)
        data_id = data_info['did']
    elif data_id is not None:
        if version != 'active':
            raise ValueError('Dataset data_id={} and version={} passed, but you can only specify a numeric data_id or a version, not both.'.format(data_id, version))
    else:
        raise ValueError('Neither name nor data_id are provided. Please provide name or data_id.')
    data_description = _get_data_description_by_id(data_id, data_home)
    if data_description['status'] != 'active':
        warn('Version {} of dataset {} is inactive, meaning that issues have been found in the dataset. Try using a newer version from this URL: {}'.format(data_description['version'], data_description['name'], data_description['url']))
    if 'error' in data_description:
        warn('OpenML registered a problem with the dataset. It might be unusable. Error: {}'.format(data_description['error']))
    if 'warning' in data_description:
        warn('OpenML raised a warning on the dataset. It might be unusable. Warning: {}'.format(data_description['warning']))
    return_sparse = data_description['format'].lower() == 'sparse_arff'
    as_frame = not return_sparse if as_frame == 'auto' else as_frame
    if parser == 'auto':
        parser_ = 'liac-arff' if return_sparse else 'pandas'
    else:
        parser_ = parser
    if parser_ == 'pandas':
        try:
            check_pandas_support('`fetch_openml`')
        except ImportError as exc:
            if as_frame:
                err_msg = "Returning pandas objects requires pandas to be installed. Alternatively, explicitly set `as_frame=False` and `parser='liac-arff'`."
            else:
                err_msg = f"Using `parser={parser!r}` wit dense data requires pandas to be installed. Alternatively, explicitly set `parser='liac-arff'`."
            raise ImportError(err_msg) from exc
    if return_sparse:
        if as_frame:
            raise ValueError("Sparse ARFF datasets cannot be loaded with as_frame=True. Use as_frame=False or as_frame='auto' instead.")
        if parser_ == 'pandas':
            raise ValueError(f"Sparse ARFF datasets cannot be loaded with parser={parser!r}. Use parser='liac-arff' or parser='auto' instead.")
    features_list = _get_data_features(data_id, data_home)
    if not as_frame:
        for feature in features_list:
            if 'true' in (feature['is_ignore'], feature['is_row_identifier']):
                continue
            if feature['data_type'] == 'string':
                raise ValueError('STRING attributes are not supported for array representation. Try as_frame=True')
    if target_column == 'default-target':
        target_columns = [feature['name'] for feature in features_list if feature['is_target'] == 'true']
    elif isinstance(target_column, str):
        target_columns = [target_column]
    elif target_column is None:
        target_columns = []
    else:
        target_columns = target_column
    data_columns = _valid_data_column_names(features_list, target_columns)
    shape: Optional[Tuple[int, int]]
    if not return_sparse:
        data_qualities = _get_data_qualities(data_id, data_home)
        shape = (_get_num_samples(data_qualities), len(features_list))
    else:
        shape = None
    url = _DATA_FILE.format(data_description['file_id'])
    bunch = _download_data_to_bunch(url, return_sparse, data_home, as_frame=bool(as_frame), openml_columns_info=features_list, shape=shape, target_columns=target_columns, data_columns=data_columns, md5_checksum=data_description['md5_checksum'], n_retries=n_retries, delay=delay, parser=parser_, read_csv_kwargs=read_csv_kwargs)
    if return_X_y:
        return (bunch.data, bunch.target)
    description = '{}\n\nDownloaded from openml.org.'.format(data_description.pop('description'))
    bunch.update(DESCR=description, details=data_description, url='https://www.openml.org/d/{}'.format(data_id))
    return bunch