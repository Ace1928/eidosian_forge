from __future__ import annotations
import itertools
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs.hashing import hash_object_array
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
def hash_pandas_object(obj: Index | DataFrame | Series, index: bool=True, encoding: str='utf8', hash_key: str | None=_default_hash_key, categorize: bool=True) -> Series:
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    obj : Index, Series, or DataFrame
    index : bool, default True
        Include the index in the hash (if Series/DataFrame).
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    Series of uint64, same length as the object

    Examples
    --------
    >>> pd.util.hash_pandas_object(pd.Series([1, 2, 3]))
    0    14639053686158035780
    1     3869563279212530728
    2      393322362522515241
    dtype: uint64
    """
    from pandas import Series
    if hash_key is None:
        hash_key = _default_hash_key
    if isinstance(obj, ABCMultiIndex):
        return Series(hash_tuples(obj, encoding, hash_key), dtype='uint64', copy=False)
    elif isinstance(obj, ABCIndex):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype('uint64', copy=False)
        ser = Series(h, index=obj, dtype='uint64', copy=False)
    elif isinstance(obj, ABCSeries):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype('uint64', copy=False)
        if index:
            index_iter = (hash_pandas_object(obj.index, index=False, encoding=encoding, hash_key=hash_key, categorize=categorize)._values for _ in [None])
            arrays = itertools.chain([h], index_iter)
            h = combine_hash_arrays(arrays, 2)
        ser = Series(h, index=obj.index, dtype='uint64', copy=False)
    elif isinstance(obj, ABCDataFrame):
        hashes = (hash_array(series._values, encoding, hash_key, categorize) for _, series in obj.items())
        num_items = len(obj.columns)
        if index:
            index_hash_generator = (hash_pandas_object(obj.index, index=False, encoding=encoding, hash_key=hash_key, categorize=categorize)._values for _ in [None])
            num_items += 1
            _hashes = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)
        h = combine_hash_arrays(hashes, num_items)
        ser = Series(h, index=obj.index, dtype='uint64', copy=False)
    else:
        raise TypeError(f'Unexpected type for hashing {type(obj)}')
    return ser