from __future__ import annotations
from collections import defaultdict
from collections.abc import (
import itertools
from typing import (
import numpy as np
from pandas._libs.sparse import IntIndex
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays import SparseArray
from pandas.core.arrays.categorical import factorize_from_iterable
from pandas.core.arrays.string_ import StringDtype
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import (
from pandas.core.series import Series
def from_dummies(data: DataFrame, sep: None | str=None, default_category: None | Hashable | dict[str, Hashable]=None) -> DataFrame:
    """
    Create a categorical ``DataFrame`` from a ``DataFrame`` of dummy variables.

    Inverts the operation performed by :func:`~pandas.get_dummies`.

    .. versionadded:: 1.5.0

    Parameters
    ----------
    data : DataFrame
        Data which contains dummy-coded variables in form of integer columns of
        1's and 0's.
    sep : str, default None
        Separator used in the column names of the dummy categories they are
        character indicating the separation of the categorical names from the prefixes.
        For example, if your column names are 'prefix_A' and 'prefix_B',
        you can strip the underscore by specifying sep='_'.
    default_category : None, Hashable or dict of Hashables, default None
        The default category is the implied category when a value has none of the
        listed categories specified with a one, i.e. if all dummies in a row are
        zero. Can be a single value for all variables or a dict directly mapping
        the default categories to a prefix of a variable.

    Returns
    -------
    DataFrame
        Categorical data decoded from the dummy input-data.

    Raises
    ------
    ValueError
        * When the input ``DataFrame`` ``data`` contains NA values.
        * When the input ``DataFrame`` ``data`` contains column names with separators
          that do not match the separator specified with ``sep``.
        * When a ``dict`` passed to ``default_category`` does not include an implied
          category for each prefix.
        * When a value in ``data`` has more than one category assigned to it.
        * When ``default_category=None`` and a value in ``data`` has no category
          assigned to it.
    TypeError
        * When the input ``data`` is not of type ``DataFrame``.
        * When the input ``DataFrame`` ``data`` contains non-dummy data.
        * When the passed ``sep`` is of a wrong data type.
        * When the passed ``default_category`` is of a wrong data type.

    See Also
    --------
    :func:`~pandas.get_dummies` : Convert ``Series`` or ``DataFrame`` to dummy codes.
    :class:`~pandas.Categorical` : Represent a categorical variable in classic.

    Notes
    -----
    The columns of the passed dummy data should only include 1's and 0's,
    or boolean values.

    Examples
    --------
    >>> df = pd.DataFrame({"a": [1, 0, 0, 1], "b": [0, 1, 0, 0],
    ...                    "c": [0, 0, 1, 0]})

    >>> df
       a  b  c
    0  1  0  0
    1  0  1  0
    2  0  0  1
    3  1  0  0

    >>> pd.from_dummies(df)
    0     a
    1     b
    2     c
    3     a

    >>> df = pd.DataFrame({"col1_a": [1, 0, 1], "col1_b": [0, 1, 0],
    ...                    "col2_a": [0, 1, 0], "col2_b": [1, 0, 0],
    ...                    "col2_c": [0, 0, 1]})

    >>> df
          col1_a  col1_b  col2_a  col2_b  col2_c
    0       1       0       0       1       0
    1       0       1       1       0       0
    2       1       0       0       0       1

    >>> pd.from_dummies(df, sep="_")
        col1    col2
    0    a       b
    1    b       a
    2    a       c

    >>> df = pd.DataFrame({"col1_a": [1, 0, 0], "col1_b": [0, 1, 0],
    ...                    "col2_a": [0, 1, 0], "col2_b": [1, 0, 0],
    ...                    "col2_c": [0, 0, 0]})

    >>> df
          col1_a  col1_b  col2_a  col2_b  col2_c
    0       1       0       0       1       0
    1       0       1       1       0       0
    2       0       0       0       0       0

    >>> pd.from_dummies(df, sep="_", default_category={"col1": "d", "col2": "e"})
        col1    col2
    0    a       b
    1    b       a
    2    d       e
    """
    from pandas.core.reshape.concat import concat
    if not isinstance(data, DataFrame):
        raise TypeError(f"Expected 'data' to be a 'DataFrame'; Received 'data' of type: {type(data).__name__}")
    col_isna_mask = cast(Series, data.isna().any())
    if col_isna_mask.any():
        raise ValueError(f"Dummy DataFrame contains NA value in column: '{col_isna_mask.idxmax()}'")
    try:
        data_to_decode = data.astype('boolean', copy=False)
    except TypeError:
        raise TypeError('Passed DataFrame contains non-dummy data')
    variables_slice = defaultdict(list)
    if sep is None:
        variables_slice[''] = list(data.columns)
    elif isinstance(sep, str):
        for col in data_to_decode.columns:
            prefix = col.split(sep)[0]
            if len(prefix) == len(col):
                raise ValueError(f'Separator not specified for column: {col}')
            variables_slice[prefix].append(col)
    else:
        raise TypeError(f"Expected 'sep' to be of type 'str' or 'None'; Received 'sep' of type: {type(sep).__name__}")
    if default_category is not None:
        if isinstance(default_category, dict):
            if not len(default_category) == len(variables_slice):
                len_msg = f"Length of 'default_category' ({len(default_category)}) did not match the length of the columns being encoded ({len(variables_slice)})"
                raise ValueError(len_msg)
        elif isinstance(default_category, Hashable):
            default_category = dict(zip(variables_slice, [default_category] * len(variables_slice)))
        else:
            raise TypeError(f"Expected 'default_category' to be of type 'None', 'Hashable', or 'dict'; Received 'default_category' of type: {type(default_category).__name__}")
    cat_data = {}
    for prefix, prefix_slice in variables_slice.items():
        if sep is None:
            cats = prefix_slice.copy()
        else:
            cats = [col[len(prefix + sep):] for col in prefix_slice]
        assigned = data_to_decode.loc[:, prefix_slice].sum(axis=1)
        if any(assigned > 1):
            raise ValueError(f'Dummy DataFrame contains multi-assignment(s); First instance in row: {assigned.idxmax()}')
        if any(assigned == 0):
            if isinstance(default_category, dict):
                cats.append(default_category[prefix])
            else:
                raise ValueError(f'Dummy DataFrame contains unassigned value(s); First instance in row: {assigned.idxmin()}')
            data_slice = concat((data_to_decode.loc[:, prefix_slice], assigned == 0), axis=1)
        else:
            data_slice = data_to_decode.loc[:, prefix_slice]
        cats_array = data._constructor_sliced(cats, dtype=data.columns.dtype)
        true_values = data_slice.idxmax(axis=1)
        indexer = data_slice.columns.get_indexer_for(true_values)
        cat_data[prefix] = cats_array.take(indexer).set_axis(data.index)
    result = DataFrame(cat_data)
    if sep is not None:
        result.columns = result.columns.astype(data.columns.dtype)
    return result